from typing import Union

import tensorflow as tf

from angorapy.agent.core import extract_discrete_action_probabilities
from angorapy.agent.ppo import loss
from angorapy.common.policies import BasePolicyDistribution
from angorapy.common.senses import Sensation


def learn_on_batch(batch,
                   joint: tf.keras.Model,
                   distribution: BasePolicyDistribution,
                   continuous_control: bool,
                   clip_values: bool,
                   clipping_bound: tf.Tensor,
                   gradient_clipping: Union[tf.Tensor, None],
                   c_value: tf.Tensor,
                   c_entropy: tf.Tensor,
                   is_recurrent: bool):
    """Optimize a given network on the given batch.

    Note:
        - the model must be given as a parameter because otherwise the tf.function wrapper will permanently
        associate the network variables with the graph, preventing us from clearing the memory

    Args:
        batch:                  the batch of data to learn on
        joint:                  the network (with both a policy and a value head
        distribution:           the distribution the network predicts
        continuous_control:     whether the distribution is continuous
        clip_values:            whether the value function should be clipped
        clipping_bound:         the bounds of the clipping
        gradient_clipping:      the clipping bound of the gradients, if None, no clipping is performed
        c_value:                the weight of the value functions loss
        c_entropy:              the weight of the entropy regularization
        is_recurrent:           whether the network is recurrent

    Returns:
        gradients, mean_entropym , mean_policy_loss, mean_value_loss
    """
    with tf.GradientTape() as tape:
        state_batch = {fname: f for fname, f in batch.items() if fname in Sensation.sense_names}
        old_values = batch["value"]

        policy_output, value_output = joint(state_batch, training=True)

        if continuous_control:
            # if action space is continuous, calculate PDF at chosen action value
            for moment in policy_output:
                tf.debugging.assert_all_finite(moment, "A moment in policy output is nan/inf")

            action_probabilities = distribution.log_probability(batch["action"], *policy_output)
        else:
            # if the action space is discrete, extract the probabilities of actions actually chosen
            action_probabilities = distribution.log_probability(batch["action"], policy_output)

        # calculate the three loss components
        policy_loss = loss.policy_loss(action_prob=action_probabilities,
                                       old_action_prob=batch["action_prob"],
                                       advantage=batch["advantage"],
                                       mask=batch["mask"],
                                       clipping_bound=clipping_bound,
                                       is_recurrent=is_recurrent)
        value_loss = loss.value_loss(value_predictions=tf.squeeze(value_output, axis=-1),
                                     old_values=old_values,
                                     returns=batch["return"],
                                     mask=batch["mask"],
                                     clip=clip_values,
                                     clipping_bound=clipping_bound,
                                     is_recurrent=is_recurrent)
        entropy = loss.entropy_bonus(policy_output=policy_output,
                                     distribution=distribution)

        # combine weighted losses
        total_loss = policy_loss + tf.multiply(c_value, value_loss) - tf.multiply(c_entropy, entropy)

    # calculate the gradient of the joint model based on total loss
    gradients = tape.gradient(total_loss, joint.trainable_variables)

    for i, gradient in enumerate(gradients):
        tf.debugging.assert_all_finite(gradient, f"Gradient {i} is nan/inf")

    # clip gradients to avoid gradient explosion and stabilize learning
    if gradient_clipping is not None:
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_clipping)

    entropy, policy_loss, value_loss = tf.reduce_mean(entropy), tf.reduce_mean(policy_loss), tf.reduce_mean(value_loss)

    return gradients, entropy, policy_loss, value_loss


def learn_on_batch_with_var(batch,
                            joint: tf.keras.Model,
                            distribution: BasePolicyDistribution,
                            continuous_control: bool,
                            clip_values: bool,
                            clipping_bound: tf.Tensor,
                            gradient_clipping: Union[tf.Tensor, None],
                            c_value: tf.Tensor,
                            c_entropy: tf.Tensor,
                            c_var: tf.Tensor,
                            var_pred: bool,
                            var_by_adv: bool,
                            is_recurrent: bool):
    """Optimize a given network on the given batch.

    Note:
        - the model must be given as a parameter because otherwise the tf.function wrapper will permanently
        associate the network variables with the graph, preventing us from clearing the memory

    Args:
        batch:                  the batch of data to learn on
        joint:                  the network (with both a policy and a value head
        distribution:           the distribution the network predicts
        continuous_control:     whether the distribution is continuous
        clip_values:            whether the value function should be clipped
        clipping_bound:         the bounds of the clipping
        gradient_clipping:      the clipping bound of the gradients, if None, no clipping is performed
        c_value:                the weight of the value functions loss
        c_entropy:              the weight of the entropy regularization
        c_pseudo_variance       not implemented yet
        is_recurrent:           whether the network is recurrent

    Returns:
        gradients, mean_entropym , mean_policy_loss, mean_value_loss
    """
    with tf.GradientTape() as tape:
        state_batch = {fname: f for fname, f in batch.items() if fname in Sensation.sense_names}
        old_values = batch["value"]

        if var_pred:
            policy_output, pseudo_var_output ,value_output = joint(state_batch, training=True)
        else:
            policy_output ,value_output = joint(state_batch, training=True)
            pseudo_var_output = [0] * len(value_output)

        if continuous_control:
            # if action space is continuous, calculate PDF at chosen action value
            for moment in policy_output:
                tf.debugging.assert_all_finite(moment, "A moment in policy output is nan/inf")

            action_probabilities = distribution.log_probability(batch["action"], *policy_output)
        else:
            # if the action space is discrete, extract the probabilities of actions actually chosen
            action_probabilities = distribution.log_probability(batch["action"], policy_output)

        # calculate the three loss components
        policy_loss = loss.policy_loss(action_prob=action_probabilities,
                                       old_action_prob=batch["action_prob"],
                                       advantage=batch["advantage"],
                                       mask=batch["mask"],
                                       clipping_bound=clipping_bound,
                                       is_recurrent=is_recurrent)
        value_loss = loss.value_loss(value_predictions=tf.squeeze(value_output, axis=-1),
                                     old_values=old_values,
                                     returns=batch["return"],
                                     mask=batch["mask"],
                                     clip=clip_values,
                                     clipping_bound=clipping_bound,
                                     is_recurrent=is_recurrent)
        entropy = loss.entropy_bonus(policy_output=policy_output,
                                     distribution=distribution)
        pseudo_variance_loss = loss.pseudo_var_loss(pseudo_var_predictions=pseudo_var_output,
                                    old_pseudo_var= batch["variance_preds"],
                                    true_pseudo_var=batch["pseudo_variance"],
                                    var_pred=var_pred,
                                    mask= batch["mask"],
                                    clip = clip_values,
                                    clipping_bound= clipping_bound,
                                    is_recurrent=is_recurrent)
        
        # combine weighted losses
        var_term = tf.multiply(c_var, pseudo_variance_loss)

        if var_by_adv:
            advantage = tf.reduce_mean(batch["advantage"])
            var_term = tf.divide(var_term, advantage)
  
        total_loss = policy_loss + tf.multiply(c_value, value_loss) - tf.multiply(c_entropy, entropy) + var_term

    # calculate the gradient of the joint model based on total loss
    gradients = tape.gradient(total_loss, joint.trainable_variables)

    for i, gradient in enumerate(gradients):
        tf.debugging.assert_all_finite(gradient, f"Gradient {i} is nan/inf")

    # clip gradients to avoid gradient explosion and stabilize learning
    if gradient_clipping is not None:
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_clipping)

    entropy, policy_loss, value_loss = tf.reduce_mean(entropy), tf.reduce_mean(policy_loss), tf.reduce_mean(value_loss)

    return gradients, entropy, policy_loss, value_loss