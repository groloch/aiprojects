from . import convolution_block, depthwise_separable_convoltion_block, dense_block
from keras.layers import Input, UpSampling2D, Activation, GlobalAveragePooling2D
from keras.layers import Concatenate, concatenate
from keras.models import Model


def create_generator() -> Model:
    input_latent = x1 = Input((10,), name='input_latent') # Random variable

    x = dense_block(units=7*7*32, do_bn=False,
                    target_shape=(7, 7, 32), name='latent_embedding')(x1)
    x = convolution_block(32, (3, 3))(x)
    x = depthwise_separable_convoltion_block(32, (3, 3))(x)

    x = UpSampling2D()(x)
    x = convolution_block(64, (3, 3))(x)
    x = depthwise_separable_convoltion_block(64, (3, 3))(x)

    x = UpSampling2D()(x)
    x = convolution_block(128, (3, 3))(x)
    x = depthwise_separable_convoltion_block(128, (3, 3))(x)
    x = convolution_block(128, (3, 3))(x)
    x = convolution_block(1, (1, 1), do_relu=False)(x)
    x = Activation('tanh')(x)

    return Model(inputs=input_latent, outputs=x, name='mnist_generator')


def create_discriminator() -> Model:
    input_image = x = Input((28, 28, 1), name='input_image')
    x = convolution_block(32, (1, 1))(x)
    x = depthwise_separable_convoltion_block(32, (3, 3))(x)
    x = convolution_block(32, (2, 2), strides=(2, 2))(x)
    
    x = depthwise_separable_convoltion_block(64, (3, 3))(x)
    x = convolution_block(64, (2, 2), strides=(2, 2))(x)

    x = depthwise_separable_convoltion_block(128, (3, 3))(x)
    x = convolution_block(128, (2, 2))(x)

    x = GlobalAveragePooling2D()(x)

    x = dense_block(1, do_relu=False, do_bn=False)(x)

    return Model(inputs=input_image, outputs=x, name='mnist_discriminator')