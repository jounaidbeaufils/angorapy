#!/usr/bin/env python
"""Functions for gathering experience and communicating it to the main thread."""
import abc
import gc
import random
from typing import Tuple, Any

import numpy as np
import tensorflow as tf
from gym.spaces import Box

from angorapy.agent.core import estimate_episode_advantages
from angorapy.agent.dataio import tf_serialize_example, make_dataset_and_stats, serialize_sample, make_dataset_and_stats_with_var
from angorapy.common.data_buffers import ExperienceBuffer, VarExperienceBuffer, TimeSequenceExperienceBuffer
from angorapy.common.policies import BasePolicyDistribution
from angorapy.common.senses import Sensation
from angorapy.common.wrappers import BaseWrapper, make_env
from angorapy.common.const import STORAGE_DIR, DETERMINISTIC
from angorapy.utilities.datatypes import StatBundle
from angorapy.utilities.model_utils import is_recurrent_model
from angorapy.utilities.util import add_state_dims, flatten, env_extract_dims

import angorapy.agent.variance as variance

class BaseGatherer(abc.ABC):

    @abc.abstractmethod
    def collect(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def postprocess(self, buffer: ExperienceBuffer, model: tf.keras.Model, env: BaseWrapper) -> ExperienceBuffer:
        pass

    @abc.abstractmethod
    def select_action(self, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        pass


class Gatherer(BaseGatherer):
    """Standard worker implementation for collecting experience by rolling out a policy.

    This is the default PPO behaviour.
    """

    def __init__(
            self,
            worker_id: int,
            exp_id: int,
            distribution: BasePolicyDistribution, 
            horizon: int,
            discount: float,
            lam: float,
            subseq_length: int):
        """

        Args:
            worker_id:
            exp_id:
            horizon:
            lam:
            discount:
            distribution:   policy distribution object
        """
        self.worker_id = worker_id
        self.exp_id = exp_id

        # parameters
        self.distribution = distribution
        self.horizon = horizon
        self.lam = lam
        self.discount = discount
        self.subseq_length = subseq_length

    def collect(self,
                joint: tf.keras.Model,
                env: BaseWrapper,
                collector_id: int) -> StatBundle:
        """Collect a batch shard of experience for a given number of time steps.

        Args:
            joint:          network returning both policy and value
            env:            environment from which to gather the data
            horizon:        the number of steps gatherd by this worker
            discount:       discount factor
            lam:            lambda parameter of GAE balancing the tradeoff between bias and variance
            subseq_length:  the length of connected subsequences for TBPTT
            collector_id:   the ID of this gathering, different from the worker's ID
        """
        state: Sensation

        is_recurrent = is_recurrent_model(joint)
        is_continuous = isinstance(env.action_space, Box)
        state_dim, action_dim = env_extract_dims(env)

        if DETERMINISTIC:
            env.seed(1)

        # reset states of potentially recurrent net
        if is_recurrent:
            joint.reset_states()

        # buffer storing the experience and stats
        if is_recurrent:
            assert self.horizon % self.subseq_length == 0, "Subsequence length would require cutting of part of the n_steps."
            buffer = TimeSequenceExperienceBuffer(self.horizon // self.subseq_length, state_dim, action_dim,
                                                  is_continuous, self.subseq_length)
        else:
            buffer = ExperienceBuffer(self.horizon, state_dim, action_dim, is_continuous)

        # go for it
        t, current_episode_return, episode_steps, current_subseq_length = 0, 0, 1, 0
        states, rewards, actions, action_probabilities, values, advantages, dones = [], [], [], [], [], [], []
        episode_endpoints = []
        achieved_goals = []
        state = env.reset()

        while t < self.horizon:
            current_subseq_length += 1

            # based on given state, predict action distribution and state value; need flatten due to tf eager bug
            prepared_state = state.with_leading_dims(time=is_recurrent).dict_as_tf()
            policy_out = flatten(joint(prepared_state))

            predicted_distribution_parameters, value = policy_out[:-1], policy_out[-1]

            # from the action distribution sample an action and remember both the action and its probability
            action, action_probability = self.select_action(predicted_distribution_parameters)

            states.append(state)
            values.append(np.squeeze(value))
            actions.append(action)
            action_probabilities.append(action_probability)  # should probably ensure that no probability is ever 0

            # make a step based on the chosen action and collect the reward for this state
            observation, reward, done, info = env.step(np.atleast_1d(action) if is_continuous else action)
            current_episode_return += (reward if "original_reward" not in info else info["original_reward"])
            rewards.append(reward)
            dones.append(done)

            if hasattr(info, "keys") and "achieved_goal" in info.keys():
                achieved_goals.append(info["achieved_goal"])

            # if recurrent, at a subsequence breakpoint/episode end stack the n_steps and buffer them
            if is_recurrent and (current_subseq_length == self.subseq_length or done):
                buffer.push_seq_to_buffer(states=states,
                                          actions=actions,
                                          action_probabilities=action_probabilities,
                                          values=values[-current_subseq_length:],
                                          episode_ended=done)

                # clear the buffered information
                states, actions, action_probabilities = [], [], []
                current_subseq_length = 0

            # depending on whether the state is terminal, choose the next state
            if done:
                episode_endpoints.append(t)

                # calculate advantages for the finished episode, where the last value is 0 since it refers to the
                # terminal state that we just observed
                episode_advantages = estimate_episode_advantages(rewards[-episode_steps:],
                                                                 values[-episode_steps:] + [0],
                                                                 self.discount, self.lam)
                episode_returns = episode_advantages + values[-episode_steps:]

                if is_recurrent:
                    # skip as many steps as are missing to fill the subsequence, then push adv ant ret to buffer
                    t += self.subseq_length - (t % self.subseq_length) - 1
                    buffer.push_adv_ret_to_buffer(episode_advantages, episode_returns)
                else:
                    advantages.append(episode_advantages)

                # reset environment to receive next episodes initial state
                state = env.reset()

                if is_recurrent:
                    joint.reset_states()

                # update/reset some statistics and trackers
                buffer.episode_lengths.append(episode_steps)
                buffer.episode_rewards.append(current_episode_return)
                buffer.episodes_completed += 1
                episode_steps = 1
                current_episode_return = 0
            else:
                state = observation
                episode_steps += 1

            t += 1

        # get last non-visited state value to incorporate it into the advantage estimation of last visited state
        values.append(np.squeeze(joint(add_state_dims(state, dims=2 if is_recurrent else 1).dict())[-1]))

        # if there was at least one step in the environment after the last episode end, calculate advantages for them
        if episode_steps > 1:
            leftover_advantages = estimate_episode_advantages(rewards[-episode_steps + 1:], values[-episode_steps:],
                                                              self.discount, self.lam)
            if is_recurrent:
                leftover_returns = leftover_advantages + values[-len(leftover_advantages) - 1:-1]
                buffer.push_adv_ret_to_buffer(leftover_advantages, leftover_returns)
            else:
                advantages.append(leftover_advantages)

        # if not recurrent, fill the buffer with everything we gathered
        if not is_recurrent:
            values = np.array(values, dtype="float32")

            # write to the buffer
            advantages = np.hstack(advantages).astype("float32")
            returns = advantages + values[:-1]
            buffer.fill(states,
                        np.array(actions, dtype="float32" if is_continuous else "int32"),
                        np.array(action_probabilities, dtype="float32"),
                        advantages,
                        returns,
                        values[:-1],
                        np.array(dones),
                        np.array(achieved_goals))

        # postprocessing steps
        buffer = self.postprocess(buffer, joint, env)

        # convert buffer to dataset and save it to tf record
        dataset, stats = make_dataset_and_stats(buffer)
        with tf.io.TFRecordWriter(f"{STORAGE_DIR}/{self.exp_id}_data_{collector_id}.tfrecord") as file_writer:
            feature_names = ([sense for sense in Sensation.sense_names if sense in observation]
                             + ["action", "action_prob", "return", "advantage", "value", "done", "mask"])

            for batch in dataset:
                inputs = [batch[f] for f in feature_names]
                record = serialize_sample(*inputs, feature_names=feature_names)

                file_writer.write(record)

        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        return stats

    def postprocess(self, buffer: ExperienceBuffer, model: tf.keras.Model, env: BaseWrapper):
        """Postprocess the gathered data."""
        buffer.normalize_advantages()

        return buffer

    def select_action(self, predicted_parameters: list) -> Tuple[tf.Tensor, np.ndarray]:
        """Standard action selection where an action is sampled fully from the predicted distribution."""
        action, action_probability = self.distribution.act(*predicted_parameters)
        action = action if not DETERMINISTIC else np.zeros(action.shape)

        return action, action_probability

class VarGatherer(Gatherer):
    """Var worker implementation for collecting experience by rolling out a policy."""

    # define __init__() overide to set default variance stragy
    def __init__(self, worker_id: int, exp_id: int, distribution: BasePolicyDistribution, horizon: int, var_discount: float, discount: float, lam: float, subseq_length: int):
        super().__init__(worker_id, exp_id, distribution, horizon, discount, lam, subseq_length)
        self.var_strategy = variance.estimate_episode_variance
        self.var_discount = var_discount


    def collect(self,
                joint: tf.keras.Model,
                env: BaseWrapper,
                collector_id: int) -> StatBundle:
        """Collect a batch shard of experience for a given number of time steps.

        Args:
            joint:          network returning both policy and value
            env:            environment from which to gather the data
            horizon:        the number of steps gatherd by this worker
            discount:       discount factor
            lam:            lambda parameter of GAE balancing the tradeoff between bias and variance
            subseq_length:  the length of connected subsequences for TBPTT
            collector_id:   the ID of this gathering, different from the worker's ID
        """
        state: Sensation

        is_recurrent = is_recurrent_model(joint)
        is_continuous = isinstance(env.action_space, Box)
        state_dim, action_dim = env_extract_dims(env)

        if DETERMINISTIC:
            env.seed(1)

        # reset states of potentially recurrent net
        if is_recurrent:

            raise NotImplementedError("pseudo_variance not available in recurrent networks")

            joint.reset_states()

        # buffer storing the experience and stats
        if is_recurrent:
            assert self.horizon % self.subseq_length == 0, "Subsequence length would require cutting of part of the n_steps."
            buffer = TimeSequenceExperienceBuffer(self.horizon // self.subseq_length, state_dim, action_dim,
                                                  is_continuous, self.subseq_length)
        else:
            buffer = VarExperienceBuffer(self.horizon, state_dim, action_dim, is_continuous)

        # go for it
        t, current_episode_return, episode_steps, current_subseq_length = 0, 0, 1, 0
        states, rewards, actions, action_probabilities, values, variance_preds, advantages, pseudo_variances, dones = [], [], [], [], [], [], [], [], []        
        episode_endpoints = []
        achieved_goals = []
        state = env.reset()

        while t < self.horizon:
            current_subseq_length += 1

            # based on given state, predict action distribution and state value; need flatten due to tf eager bug
            prepared_state = state.with_leading_dims(time=is_recurrent).dict_as_tf()
            policy_out = flatten(joint(prepared_state))

            if len(joint.output) == 2: #assume using no_preds (yes this bad code)
                predicted_distribution_parameters, value = policy_out[:-1], policy_out[-1]
                variance_pred = [0] * len(value)
            else:
                predicted_distribution_parameters, variance_pred ,value = policy_out[:-2], policy_out[-2], policy_out[-1]

            # from the action distribution sample an action and remember both the action and its probability
            action, action_probability = self.select_action(predicted_distribution_parameters)

            states.append(state)
            values.append(np.squeeze(value))
            variance_preds.extend(variance_pred)
            actions.append(action)
            action_probabilities.append(action_probability)  # should probably ensure that no probability is ever 0

            # make a step based on the chosen action and collect the reward for this state
            observation, reward, done, info = env.step(np.atleast_1d(action) if is_continuous else action)
            current_episode_return += (reward if "original_reward" not in info else info["original_reward"])
            rewards.append(reward)
            dones.append(done)

            if hasattr(info, "keys") and "achieved_goal" in info.keys():
                achieved_goals.append(info["achieved_goal"])

            # if recurrent, at a subsequence breakpoint/episode end stack the n_steps and buffer them
            if is_recurrent and (current_subseq_length == self.subseq_length or done):
                buffer.push_seq_to_buffer(states=states,
                                          actions=actions,
                                          action_probabilities=action_probabilities,
                                          values=values[-current_subseq_length:],
                                          episode_ended=done)

                # clear the buffered information
                states, actions, action_probabilities = [], [], []
                current_subseq_length = 0

            # depending on whether the state is terminal, choose the next state
            if done:
                episode_endpoints.append(t)

                # calculate advantages for the finished episode, where the last value is 0 since it refers to the
                # terminal state that we just observed
                episode_advantages = estimate_episode_advantages(rewards[-episode_steps:],
                                                                 values[-episode_steps:] + [0],
                                                                 self.discount, self.lam)
                episode_returns = episode_advantages + values[-episode_steps:]

                # calculate pseudo variance for the finished episode (Jounaid)
                episode_pseudo_variances = self.var_strategy(rewards[-episode_steps:],
                                                                 variance_preds[-episode_steps:] + [(0, 1)],
                                                                 self.var_discount, 
                                                                 self.lam)

                if is_recurrent:
                    # skip as many steps as are missing to fill the subsequence, then push adv ant ret to buffer
                    t += self.subseq_length - (t % self.subseq_length) - 1
                    buffer.push_adv_ret_to_buffer(episode_advantages, episode_returns)
                else:
                    advantages.append(episode_advantages)
                    # (Jounaid)
                    pseudo_variances.extend(episode_pseudo_variances)

                # reset environment to receive next episodes initial state
                state = env.reset()

                if is_recurrent:
                    joint.reset_states()

                # update/reset some statistics and trackers
                buffer.episode_lengths.append(episode_steps)
                buffer.episode_rewards.append(current_episode_return)
                buffer.episodes_completed += 1
                episode_steps = 1
                current_episode_return = 0
            else:
                state = observation
                episode_steps += 1

            t += 1

        # get last non-visited state value to incorporate it into the advantage estimation of last visited state
        values.append(np.squeeze(joint(add_state_dims(state, dims=2 if is_recurrent else 1).dict())[-1]))

        if len(joint.output) == 2: #assume using VarGathererAbs subclass (yes this bad code)
                variance_preds.extend([0])
        else:
            variance_preds.extend(joint(add_state_dims(state, dims=2 if is_recurrent else 1).dict())[-2])

        # if there was at least one step in the environment after the last episode end, calculate advantages for them
        if episode_steps > 1:
            leftover_advantages = estimate_episode_advantages(rewards[-episode_steps + 1:],
                                                              values[-episode_steps:],
                                                              self.discount, self.lam)
            # (Jounaid)
            leftover_pseudo_variance = self.var_strategy(rewards[-episode_steps + 1:],
                                                                 variance_preds[-episode_steps:],
                                                                 self.var_discount, 
                                                                 self.lam)
            if is_recurrent:
                leftover_returns = leftover_advantages + values[-len(leftover_advantages) - 1:-1]
                buffer.push_adv_ret_to_buffer(leftover_advantages, leftover_returns)
            else:
                advantages.append(leftover_advantages)
                # (Jounaid)
                pseudo_variances.extend(leftover_pseudo_variance)

        # if not recurrent, fill the buffer with everything we gathered
        if not is_recurrent:
            values = np.array(values, dtype="float32")

            # (Jounaid)
            variance_preds = np.array(variance_preds, dtype="float32")

            # write to the buffer
            advantages = np.hstack(advantages).astype("float32")
            returns = advantages + values[:-1]
            # (Jounaid)
            pseudo_variances = np.array(pseudo_variances).astype("float32")

            buffer.fill(states,
                        np.array(actions, dtype="float32" if is_continuous else "int32"),
                        np.array(action_probabilities, dtype="float32"),
                        advantages,
                        # (Jounaid)
                        pseudo_variances,
                        returns,
                        values[:-1],
                        # (Jounaid)
                        variance_preds[:-1],
                        np.array(dones),
                        np.array(achieved_goals))

        # postprocessing steps
        buffer = self.postprocess(buffer, joint, env)

        # convert buffer to dataset and save it to tf record
        dataset, stats = make_dataset_and_stats_with_var(buffer)
        with tf.io.TFRecordWriter(f"{STORAGE_DIR}/{self.exp_id}_data_{collector_id}.tfrecord") as file_writer:
            feature_names = ([sense for sense in Sensation.sense_names if sense in observation]
                             + ["action", "action_prob", "return", "advantage", "pseudo_variance","value", "variance_preds", "done", "mask"])

            for batch in dataset:
                inputs = [batch[f] for f in feature_names]
                record = serialize_sample(*inputs, feature_names=feature_names)

                file_writer.write(record)

        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        return stats

class VarGathererAbs(VarGatherer):
    def __init__(self, worker_id: int, exp_id: int, distribution: BasePolicyDistribution, horizon: int, var_discount: float, discount: float, lam: float, subseq_length: int):
        super().__init__(worker_id, exp_id, distribution, horizon, var_discount, discount, lam, subseq_length)
        self.var_strategy = variance.absolute

class VarGathererNoPreds(VarGatherer):
    def __init__(self, worker_id: int, exp_id: int, distribution: BasePolicyDistribution, horizon: int, var_discount: float, discount: float, lam: float, subseq_length: int):
        super().__init__(worker_id, exp_id, distribution, horizon, var_discount, discount, lam, subseq_length)
        self.var_strategy = variance.future_reward_variance

class VarGathererNoise(VarGatherer):
    def __init__(self, worker_id: int, exp_id: int, distribution: BasePolicyDistribution, horizon: int, var_discount: float, discount: float, lam: float, subseq_length: int):
        super().__init__(worker_id, exp_id, distribution, horizon, var_discount, discount, lam, subseq_length)
        self.var_strategy = variance.noise

class EpsilonGreedyGatherer(Gatherer):
    """Exemplary epsilon greedy gathering strategy.

    This is not safe! At some point, the probability of a uniform sample may become zero under the current policy and
    thereby the log probability goes to negative infinity, leading to the optimization to collapse.
    """

    def select_action(self, predicted_parameters: list) -> Tuple[tf.Tensor, np.ndarray]:
        if random.random() < 0.9:
            action, action_probability = super(EpsilonGreedyGatherer, self).select_action(predicted_parameters)
        else:
            action = tf.cast(self.distribution.action_space.sample(), tf.float32)
            action_probability = self.distribution.log_probability(action, *predicted_parameters)

        return action, action_probability


def evaluate(policy: tf.keras.Model, env: BaseWrapper, distribution: BasePolicyDistribution,
             act_confidently=False) -> Tuple[int, int, Any]:
    """Evaluate one episode of the given environment following the given policy."""
    policy.reset_states()
    is_recurrent = is_recurrent_model(policy)
    is_continuous = isinstance(env.action_space, Box)

    done = False
    state = env.reset()
    cumulative_reward = 0
    steps = 0
    while not done:
        prepared_state = state.with_leading_dims(time=is_recurrent).dict_as_tf()
        probabilities = flatten(policy(prepared_state))

        if not act_confidently:
            action, _ = distribution.act(*probabilities)
        else:
            action = distribution.act_deterministic(*probabilities)
        observation, reward, done, info = env.step(np.atleast_1d(action) if is_continuous else action)
        cumulative_reward += info["original_reward"]
        observation = observation

        state = observation
        steps += 1

    eps_class = env.unwrapped.current_target_finger if hasattr(env.unwrapped, "current_target_finger") else None

    return steps, cumulative_reward, eps_class


def fake_env_step(env: BaseWrapper):
    """Return a random step imitating the given environment without actually stepping the environment."""
    return env.observation(env.observation_space.sample()), 1, False, {}


def fake_joint_output(joint):
    """Return a random output of a network without calling the network."""
    outshape = flatten(joint.output_shape)
    for i in range(len(outshape)):
        outshape[i] = [d if d is not None else 1 for d in outshape[i]]

    return np.float32(np.random.random(outshape[0])), \
           np.float32(np.random.random(outshape[1])), \
           np.float32(np.random.random(outshape[2]))


if __name__ == '__main__':
    environment = make_env("ReachAbsoluteVisual-v0")
    fake_env_step(environment)