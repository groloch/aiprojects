import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, Add
from keras.layers import DepthwiseConv2D, Dense, Reshape


def dense_block(units: int,
                do_bn: bool=True,
                dropouts: float=None,
                target_shape: tuple=None,
                activation: str | Activation='relu',
                name:str=None):
    if name is not None:
        dense_name = f'{name}_dense'
        act_name = f'{name}_act'
        bn_name = f'{name}_bn'
        dropout_name = f'{name}_dropout'
    else:
        dense_name = None
        act_name = None
        bn_name = None
        dropout_name = None
    def ret(x: tf.Tensor) -> tf.Tensor:
        x = Dense(units, use_bias=not do_bn, name=dense_name)(x)
        if do_bn:
            x = BatchNormalization(name=bn_name)(x)
        if activation is not None:
            if isinstance(activation, str):
                x = Activation(activation, name=act_name)(x)
            else:
                x = activation(x)
        if dropouts is not None:
            x = Dropout(dropouts, name=dropout_name)(x)
        if target_shape is not None:
            x = Reshape(target_shape)(x)
        return x
    return ret

def convolution_block(filters: int, 
                      filter_shape: tuple[int, int] | int, 
                      do_bn: bool=True, 
                      activation: str | Activation='relu',
                      dropouts: float=None, 
                      dilations: tuple[int, int] | int=(1, 1), 
                      strides: tuple[int, int] | int=(1, 1),
                      padding: str='same',
                      name: str=None):
    if name is not None:
        conv_name = f'{name}_conv'
        act_name = f'{name}_act'
        bn_name = f'{name}_bn'
        dropout_name = f'{name}_dropout'
    else:
        conv_name = None
        act_name = None
        bn_name = None
        dropout_name = None
    def ret(x: tf.Tensor) -> tf.Tensor:
        x = Conv2D(filters, filter_shape, strides=strides, 
                   padding=padding, dilation_rate=dilations,
                   use_bias=not do_bn, name=conv_name)(x)
        if do_bn:
            x = BatchNormalization(name=bn_name)(x)
        if activation is not None:
            if isinstance(activation, str):
                x = Activation(activation, name=act_name)(x)
            elif isinstance(activation, Activation):
                x = activation(x)
        if dropouts is not None:
            x = Dropout(dropouts, name=dropout_name)(x)
        return x
    return ret

def residual_block(filters: int, 
                   filter_shape: tuple[int, int] | int, 
                   do_bn: bool=True, 
                   dropouts: float=None, 
                   dilations: tuple[int, int] | int=(1, 1), 
                   strides: tuple[int, int] | int=(1, 1),
                   padding: str='same',
                   activation: str | Activation='relu',
                   name: str=None):
    if name is not None:
        conv_name = f'{name}_conv'
        add_name = f'{name}_add'
        act_name = f'{name}_act'
        bn_name = f'{name}_bn'
        dropout_name = f'{name}_dropout'
    else:
        conv_name = None
        add_name=None
        act_name = None
        bn_name = None
        dropout_name = None
    def ret(x: tf.Tensor) -> tf.Tensor:
        x_ = Conv2D(filters, filter_shape, strides=strides, 
                   padding=padding, dilation_rate=dilations,
                   use_bias=not do_bn, name=conv_name)(x)
        if do_bn:
            x_ = BatchNormalization(name=bn_name)(x)
        x = Add(name=add_name)((x, x_))
        if activation is not None:
            if isinstance(activation, str):
                x = Activation(activation, name=act_name)(x)
            elif isinstance(activation, Activation):
                x = activation(x)
        if dropouts is not None:
            x = Dropout(dropouts, name=dropout_name)(x)
        return x
    return ret

def depthwise_block(filter_shape: tuple[int, int] | int, 
                    do_bn: bool=True, 
                    dropouts: float=None, 
                    dilations: tuple[int, int] | int=(1, 1), 
                    strides: tuple[int, int] | int=(1, 1),
                    depth_multiplier: int=1,
                    padding: str='same',
                    activation: str | Activation='relu',
                    name: str=None):
    if name is not None:
        dw_name = f'{name}_dw'
        act_name = f'{name}_act'
        bn_name = f'{name}_bn'
        dropout_name = f'{name}_dropout'
    else:
        dw_name = None
        act_name = None
        bn_name = None
        dropout_name = None
    def ret(x: tf.Tensor) -> tf.Tensor:
        x = DepthwiseConv2D(filter_shape, strides=strides, 
                   padding=padding, dilation_rate=dilations,
                   use_bias=not do_bn, depth_multiplier=depth_multiplier,
                   name=dw_name)(x)
        if do_bn:
            x = BatchNormalization(name=bn_name)(x)
        if activation is not None:
            if isinstance(activation, str):
                x = Activation(activation, name=act_name)(x)
            elif isinstance(activation, Activation):
                x = activation(x)
        if dropouts is not None:
            x = Dropout(dropouts, dropout_name)(x)
        return x
    return ret

def depthwise_separable_convoltion_block(filters: int, 
                                         filter_shape: tuple[int, int] | int, 
                                         do_bn: bool=True, 
                                         dropouts: float=None, 
                                         dilations: tuple[int, int] | int=(1, 1), 
                                         strides: tuple[int, int] | int=(1, 1),
                                         padding: str='same',
                                         activation: str | Activation='relu',
                                         depth_multiplier: int=1,
                                         name: str=None):
    if name is not None:
        conv_name = f'{name}_convblock'
        dw_name = f'{name}_dwblock'
    else:
        conv_name = None
        dw_name = None
    def ret(x: tf.Tensor) -> tf.Tensor:
        x = depthwise_block(filter_shape=filter_shape,
                            do_bn=do_bn,
                            activation=activation,
                            strides=strides,
                            dilations=dilations,
                            depth_multiplier=depth_multiplier,
                            name=dw_name)(x)
        x = convolution_block(filters=filters, 
                              filter_shape=(1, 1),
                              do_bn=do_bn,
                              activation=activation,
                              dropouts=dropouts,
                              padding=padding,
                              name=conv_name)(x)
        return x
    return ret

def spacewise_separable_convoltion_block(filters: int, 
                                         filter_shape: tuple[int, int], 
                                         do_bn: bool=True, 
                                         dropouts: float=None, 
                                         dilations: tuple[int, int]=(1, 1), 
                                         strides: tuple[int, int]=(1, 1),
                                         padding: str='same',
                                         activation: str | Activation='relu',
                                         rank: int=4,
                                         name: str=None):
    if name is not None:
        conv1_name = f'{name}_convblock1'
        conv2_name = f'{name}_convblock2'
    else:
        conv1_name = None
        conv2_name = None
    def ret(x: tf.Tensor) -> tf.Tensor:
        x = convolution_block(filters=rank,
                              filter_shape=(filter_shape[0], 1),
                              do_bn=do_bn,
                              activation=activation,
                              padding=padding,
                              strides=(strides[0], 1),
                              dilations=(dilations[0], 1),
                              name=conv1_name)(x)
        x = convolution_block(filters=filters, 
                              filter_shape=(1, filter_shape[1]),
                              do_bn=do_bn,
                              activation=activation,
                              dropouts=dropouts,
                              padding=padding,
                              strides=(1, strides[1]),
                              dilations=(1, dilations[1]),
                              name=conv2_name)(x)
        return x
    return ret

def bottleneck_convolution_block(filters: int, 
                                 filter_shape: tuple[int, int] | int, 
                                 do_bn: bool=True, 
                                 activation: str | Activation='relu',
                                 dropouts: float=None, 
                                 dilations: tuple[int, int] | int=(1, 1), 
                                 strides: tuple[int, int] | int=(1, 1),
                                 padding: str='same',
                                 reduction: int = 4,
                                 name: str=None):
    if name is not None:
        conv1_name = f'{name}_convblock1'
        conv2_name = f'{name}_convblock2'
        conv3_name = f'{name}_convblock3'
    else:
        conv1_name = None
        conv2_name = None
        conv3_name = None
    def ret(x: tf.Tensor) -> tf.Tensor:
        x = convolution_block(filters=filters//reduction,
                              filter_shape=(1, 1),
                              do_bn=do_bn,
                              activation=activation,
                              name=conv1_name)(x)
        x = convolution_block(filters=filters//reduction,
                              filter_shape=filter_shape,
                              do_bn=do_bn,
                              activation=activation,
                              dilations=dilations,
                              strides=strides,
                              padding=padding,
                              name=conv2_name)(x)
        x = convolution_block(filters=filters,
                              filter_shape=filter_shape,
                              do_bn=do_bn,
                              activation=activation,
                              dropouts=dropouts,
                              name=conv3_name)(x)
        return x
    return ret


def xnet_depthwise_separable_convoltion_block(filters: int, 
                                              filter_shape: tuple[int, int] | int, 
                                              do_bn: bool=True, 
                                              activation: str | Activation='relu',
                                              dropouts: float=None, 
                                              dilations: tuple[int, int] | int=(1, 1), 
                                              strides: tuple[int, int] | int=(1, 1),
                                              padding: str='same',
                                              depth_multiplier: int=1,
                                              name: str=None):
    if name is not None:
        conv_name = f'{name}_convblock'
        dw_name = f'{name}_dwblock'
    else:
        conv_name = None
        dw_name = None
    def ret(x: tf.Tensor) -> tf.Tensor:
        x = convolution_block(filters=filters, 
                              filter_shape=(1, 1),
                              do_bn=do_bn,
                              activation=activation,
                              padding=padding,
                              namae=conv_name)(x)
        x = depthwise_block(filter_shape=filter_shape,
                            do_bn=do_bn,
                            activation=activation,
                            dilations=dilations,
                            strides=strides,
                            dropouts=dropouts,
                            depth_multiplier=depth_multiplier,
                            name=dw_name)(x)
        return x
    return ret
