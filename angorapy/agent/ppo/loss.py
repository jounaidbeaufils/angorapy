import sys

import tensorflow as tf


@tf.function
def policy_loss(action_prob: tf.Tensor,
                old_action_prob: tf.Tensor,
                advantage: tf.Tensor,
                mask: tf.Tensor,
                clipping_bound: tf.Tensor,
                is_recurrent: bool) -> tf.Tensor:
    """Clipped objective as given in the PPO paper. Original objective is to be maximized, but this is the negated
    objective to be minimized! In the recurrent version a mask is calculated based on 0 values in the
    old_action_prob tensor. The mask is then applied in the mean operation of the loss.

    Args:
      action_prob (tf.Tensor): the probability of the action for the state under the current policy
      old_action_prob (tf.Tensor): the probability of the action taken given by the old policy during the episode
      advantage (tf.Tensor): the advantage that taking the action gives over the estimated state value

    Returns:
      the value of the objective function

    """

    tf.debugging.assert_all_finite(action_prob, "action_prob is not all finite!")
    tf.debugging.assert_all_finite(old_action_prob, "old_action_prob is not all finite!")
    tf.debugging.assert_all_finite(advantage, "advantage is not all finite!")

    ratio = tf.exp(action_prob - old_action_prob)
    clipped = tf.maximum(
        tf.math.multiply(ratio, -advantage),
        tf.math.multiply(tf.clip_by_value(ratio, 1 - clipping_bound, 1 + clipping_bound), -advantage)
    )

    if is_recurrent:
        # build and apply a mask over the probabilities (recurrent)
        clipped_masked = tf.where(mask, clipped, 1)
        return tf.reduce_sum(clipped_masked) / tf.reduce_sum(tf.cast(mask, tf.float32))
    else:
        return tf.reduce_mean(clipped)


@tf.function
def value_loss(value_predictions: tf.Tensor,
               old_values: tf.Tensor,
               returns: tf.Tensor,
               mask: tf.Tensor,
               clip: bool,
               clipping_bound: tf.Tensor,
               is_recurrent: bool) -> tf.Tensor:
    """Loss of the critic network as squared error between the prediction and the sampled future return. In the
    recurrent case a mask is calculated based on 0 values in the old_action_prob tensor. This mask is then applied
    in the mean operation of the loss.

    Args:
      value_predictions (tf.Tensor): value prediction by the current critic network
      old_values (tf.Tensor): value prediction by the old critic network during gathering
      returns (tf.Tensor): discounted return estimation
      clip (object): (Default value = True) value loss can be clipped by same range as policy loss

    Returns:
      squared error between prediction and return
    """
    tf.debugging.assert_all_finite(value_predictions, "value_predictions is not all finite!")
    tf.debugging.assert_all_finite(old_values, "old_values is not all finite!")
    tf.debugging.assert_all_finite(returns, "returns is not all finite!")

    error = tf.square(value_predictions - returns)

    if clip:
        # clips value error to reduce variance
        clipped_values = old_values + tf.clip_by_value(value_predictions - old_values, -clipping_bound, clipping_bound)
        clipped_error = tf.square(clipped_values - returns)
        error = tf.maximum(clipped_error, error)

    if is_recurrent:
        # apply mask over the old values
        error_masked = tf.where(mask, error, 1)  # masking with tf.where because inf * 0 = nan...
        return (tf.reduce_sum(error_masked) / tf.reduce_sum(tf.cast(mask, tf.float32))) * 0.5
    else:
        return tf.reduce_mean(error) * 0.5


@tf.function
def entropy_bonus(policy_output: tf.Tensor, distribution) -> tf.Tensor:
    """Entropy of policy output acting as regularization by preventing dominance of one action. The higher the
    entropy, the less probability mass lies on a single action, which would hinder exploration. We hence reduce
    the loss by the (scaled by c_entropy) entropy to encourage a certain degree of exploration.

    Args:
      policy_output (tf.Tensor): a tensor containing (batches of) probabilities for actions in the case of discrete
        actions or (batches of) means and standard deviations for continuous control.

    Returns:
      entropy bonus
    """

    return tf.reduce_mean(distribution.entropy(policy_output))

@tf.function
def pseudo_var_loss_no_preds(pseudo_variance: tf.Tensor,
                mask: tf.Tensor,
                is_recurrent: bool) -> tf.Tensor:
    """pseudo variance as a measurure of reward variability.

    Args:
      pseudo_variance (tf.Tensor): the pseudo variance in reward after taking the action

    Returns:
      the value of the pseudo variance term.

    """

    tf.debugging.assert_all_finite(pseudo_variance, "pseudo variance is not all finite!")

    if is_recurrent:
        
        raise NotImplementedError("pseudo_variance not available in recurrent network")

        # build and apply a mask over the probabilities (recurrent)
        clipped_masked = tf.where(mask, pseudo_variance, 1)
        return tf.reduce_sum(clipped_masked) / tf.reduce_sum(tf.cast(mask, tf.float32))
    else:
        return tf.reduce_mean(pseudo_variance)

def pseudo_var_loss(pseudo_var_predictions: tf.Tensor,
               old_pseudo_var: tf.Tensor,
               true_pseudo_var: tf.Tensor,
               var_pred: bool,
               mask: tf.Tensor,
               clip: bool,
               clipping_bound: tf.Tensor,
               is_recurrent: bool) -> tf.Tensor:
    if var_pred:
        return variance_loss(pseudo_var_predictions,
                             old_pseudo_var,
                            true_pseudo_var,
                            mask,
                            clip,
                            clipping_bound,
                            is_recurrent)
    else:
        return pseudo_var_loss_no_preds(true_pseudo_var, mask, is_recurrent)

@tf.function
def variance_loss(pseudo_var_predictions: tf.Tensor,
               old_pseudo_var: tf.Tensor,
               true_pseudo_var: tf.Tensor,
               mask: tf.Tensor,
               clip: bool,
               clipping_bound: tf.Tensor,
               is_recurrent: bool) -> tf.Tensor:
    """Loss of the critic network as squared error between the prediction and the sampled future return. In the
    recurrent case a mask is calculated based on 0 values in the old_action_prob tensor. This mask is then applied
    in the mean operation of the loss.

    Args:
      pseudo_var_predictions (tf.Tensor): variance prediction by the current critic network
      old_pseudo_var (tf.Tensor): old variance prediction by the old critic network during gathering
      true_pseudo_var (tf.Tensor): calculated variance in episode
      clip (object): (Default value = True) value loss can be clipped by same range as policy loss

    Returns:
      squared error between prediction and return
    """
    tf.debugging.assert_all_finite(pseudo_var_predictions, "pseudo_var_predictions is not all finite!")
    tf.debugging.assert_all_finite(old_pseudo_var, "old_pseudo_var is not all finite!")
    tf.debugging.assert_all_finite(true_pseudo_var, "true_pseudo_var is not all finite!")

    # Convert list of tuples to separate lists
    var_col, n_col = tf.split(true_pseudo_var, num_or_size_splits=2, axis=1)

    # Convert lists to tensors
    var_col = tf.squeeze(var_col)
    n_col = tf.squeeze(n_col)

    # Subtract var_col from the first column of predictions and c_col from the second column of predictions
    diff = tf.subtract(pseudo_var_predictions, tf.stack([var_col, n_col], axis=1))


    error = tf.square(diff)

    if clip:
        # clips value error to reduce variance
        clipped_values = old_pseudo_var + tf.clip_by_value(pseudo_var_predictions - old_pseudo_var, -clipping_bound, clipping_bound)
        clipped_error = tf.square(clipped_values - true_pseudo_var)
        error = tf.maximum(clipped_error, error)

    if is_recurrent:
        
        raise NotImplementedError("pseudo_variance not available in recurrent network")

        # apply mask over the old values
        error_masked = tf.where(mask, error, 1)  # masking with tf.where because inf * 0 = nan...
        return (tf.reduce_sum(error_masked) / tf.reduce_sum(tf.cast(mask, tf.float32))) * 0.5
    else:
        return tf.reduce_mean(error) * 0.5