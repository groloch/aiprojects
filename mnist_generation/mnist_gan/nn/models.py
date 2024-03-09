from . import convolution_block, depthwise_separable_convoltion_block, dense_block
from keras.layers import Input, UpSampling2D, Activation, GlobalAveragePooling2D
from keras.layers import Concatenate, concatenate
from keras.models import Model


def create_generator() -> Model:
    input_latent = x1 = Input((10,), name='input_latent') # Random variable
    input_class = x2 = Input((10,), name='input_class') # One hot of the target class

    x = dense_block(units=441, do_bn=False,
                    target_shape=(7, 7, 9), name='latent_embedding')(x1)
    x2 = dense_block(units=49, do_bn=False,
                    target_shape=(7, 7, 1), name='class_embedding')(x2)
    x = Concatenate()([x, x2])
    x = convolution_block(16, (3, 3))(x)
    x = depthwise_separable_convoltion_block(16, (3, 3))(x)

    x = UpSampling2D()(x)
    x = convolution_block(32, (3, 3))(x)
    x = depthwise_separable_convoltion_block(32, (3, 3))(x)

    x = UpSampling2D()(x)
    x = convolution_block(64, (3, 3))(x)
    x = depthwise_separable_convoltion_block(64, (3, 3))(x)
    x = convolution_block(32, (3, 3))(x)
    x = convolution_block(1, (1, 1), do_relu=False)(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=[input_latent, input_class], outputs=x, name='mnist_generator')


def create_discriminator() -> Model:
    input_image = x = Input((28, 28, 1), name='input_image')
    input_class = x2 = Input((10,), name='input_class')
    x2 = dense_block(28*28, target_shape=(28, 28, 1), do_bn=False,
                     name='class_embedding')(x2)
    x = Concatenate()([x, x2])
    x = convolution_block(16, (1, 1))(x)
    x = depthwise_separable_convoltion_block(16, (3, 3))(x)
    x = convolution_block(16, (2, 2), strides=(2, 2))(x)
    
    x = depthwise_separable_convoltion_block(32, (3, 3))(x)
    x = convolution_block(32, (2, 2), strides=(2, 2))(x)

    x = depthwise_separable_convoltion_block(64, (3, 3))(x)
    x = convolution_block(64, (2, 2))(x)

    x = GlobalAveragePooling2D()(x)

    x = dense_block(1, do_relu=False)(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=[input_image, input_class], outputs=x, name='mnist_discriminator')