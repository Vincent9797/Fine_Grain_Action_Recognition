from utils import _tensor_shape

from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
from keras.layers import GlobalAveragePooling3D, Reshape, Dense, multiply, add, Permute, Conv3D
import keras.backend as K


def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block

    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters

    Returns: a Keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = _tensor_shape(init)[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([init, se])
    return x


def time_squeeze_excite_block(input_tensor, ratio=2):
    init = input_tensor
    time_axis = 1
    filters = _tensor_shape(init)[time_axis]
    se_shape = (filters, 1, 1, 1)


    se = GlobalAveragePooling3D("channels_first")(init)
    se = Reshape((filters,))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = Reshape(se_shape)(se)

    x = multiply([init, se])
    return x

def spatial_squeeze_excite_block(input_tensor):
    """ Create a spatial squeeze-excite block

    Args:
        input_tensor: input Keras tensor

    Returns: a Keras tensor

    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x


def channel_spatial_squeeze_excite(input_tensor, ratio=16):
    """ Create a spatial squeeze-excite block

    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters

    Returns: a Keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)

    x = add([cse, sse])
    return x
