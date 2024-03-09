import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras.optimizers import Optimizer, Adam


cross_entropy = BinaryCrossentropy()


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output: tf.Tensor, 
                       fake_output: tf.Tensor) -> tf.Tensor:
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_optimizer(learning_rate: float) -> Optimizer:
    return Adam(learning_rate=learning_rate)

def discriminator_optimizer(learning_rate: float) -> Optimizer:
    return Adam(learning_rate=learning_rate)
