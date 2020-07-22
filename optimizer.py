import tensorflow as tf
import numpy as np
import gin


@gin.configurable
def cosine_decay_with_warmup(learning_rate_base: float = 0.0001,
                             total_steps: int = 50000,
                             warmup_learning_rate: float = 0.0,
                             warmup_steps: int = 100,
                             hold_base_rate_steps: int = 0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
        Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
        ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Args:
        learning_rate_base: base learning rate.
        total_steps: total number of training steps.
        warmup_learning_rate: initial learning rate for warm up.
        warmup_steps: number of warmup steps.
        hold_base_rate_steps: Optional number of steps to hold base learning rate
        before decaying.
    Returns:
        If executing eagerly:
        returns a no-arg callable that outputs the (scalar)
        float tensor learning rate given the current value of global_step.
        If in a graph:
        immediately returns a (scalar) float tensor representing learning rate.
    Raises:
        ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """
    global_step = tf.compat.v1.train.get_global_step
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')

    def eager_decay_rate():
        """Callable to compute the learning rate."""
        learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
            np.pi * (tf.cast(global_step, tf.float32) - warmup_steps -
                     hold_base_rate_steps) /
            float(total_steps - warmup_steps - hold_base_rate_steps)))
        if hold_base_rate_steps > 0:
            learning_rate = tf.where(
                global_step > warmup_steps + hold_base_rate_steps,
                learning_rate, learning_rate_base)
        if warmup_steps > 0:
            if learning_rate_base < warmup_learning_rate:
                raise ValueError(
                    'learning_rate_base must be larger or equal to '
                    'warmup_learning_rate.')
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * tf.cast(global_step,
                                          tf.float32) + warmup_learning_rate
            learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate)
        return tf.where(global_step > total_steps,
                        0.0,
                        learning_rate,
                        name='learning_rate')

    if tf.executing_eagerly():
        return eager_decay_rate
    else:
        return eager_decay_rate()
