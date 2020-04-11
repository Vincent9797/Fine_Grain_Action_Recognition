"""MobileNet v3 models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""

# 2D layers
from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D

# 3D layers
from keras.layers import Conv3D, Dense, GlobalAveragePooling3D
from Group_Depthwise_Conv3D import DepthwiseConv3D
from keras.layers import Input, Activation, BatchNormalization, Add, Multiply, Reshape
from keras import backend as K
from keras.models import Model

class MobileNetBase:
    def __init__(self, shape, n_class, alpha=1.0):
        self.shape = shape
        self.n_class = n_class
        self.alpha = alpha

    def _relu6(self, x):
        return K.relu(x, max_value=6.0)

    def _hard_swish(self, x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(self, x, nl):
        if nl == 'HS':
            x = Activation(self._hard_swish)(x)
        if nl == 'RE':
            x = Activation(self._relu6)(x)

        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv3D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)

        return self._return_activation(x, nl)

    def _squeeze(self, inputs):
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling3D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        x = Reshape((1, 1, 1, input_channels))(x)
        x = Multiply()([inputs, x])

        return x

    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)

        tchannel = int(e)
        cchannel = int(self.alpha * filters)

        r = s == 1 and input_shape[4] == filters

        x = self._conv_block(inputs, tchannel, (1, 1, 1), (1, 1, 1), nl)

        x = DepthwiseConv3D(kernel, strides=(s, s, s), group_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = self._return_activation(x, nl)

        if squeeze:
            x = self._squeeze(x)

        x = Conv3D(cchannel, (1, 1, 1), strides=(1, 1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def build(self):
        pass

class MobileNetV3_Large(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        super(MobileNetV3_Large, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3, 3), strides=(2, 2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3, 3), e=16, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3, 3), e=64, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3, 3), e=72, s=1, squeeze=False, nl='RE   ')
        x = self._bottleneck(x, 40, (5, 5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 80, (3, 3, 3), e=240, s=2, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 112, (3, 3, 3), e=480, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 112, (3, 3, 3), e=672, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5, 5), e=672, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5, 5), e=960, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 960, (1, 1, 1), strides=(1, 1, 1), nl='HS')
        x = GlobalAveragePooling3D()(x)
        x = Reshape((1, 1, 1, 960))(x)

        x = Conv3D(1280, (1, 1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            if self.n_class == 1:
                x = Conv3D(self.n_class, (1, 1, 1), padding='same', activation='sigmoid')(x)
                x = Reshape((self.n_class,))(x)
            else:
                x = Conv3D(self.n_class, (1, 1, 1), padding='same', activation='softmax')(x)
                x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)

        if plot:
            plot_model(model, to_file='images/MobileNetv3_large.png', show_shapes=True)

        return model

class MobileNetV3_Small(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        super(MobileNetV3_Small, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3, 3), strides=(2, 2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 24, (3, 3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3, 3), e=88, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5, 5), e=96, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5, 5), e=144, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5, 5), e=288, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5, 5), e=576, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 576, (1, 1, 1), strides=(1, 1, 1), nl='HS')
        x = GlobalAveragePooling3D()(x)
        x = Reshape((1, 1, 1, 576))(x)

        x = Conv3D(1280, (1, 1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            if self.n_class == 1:
                x = Conv3D(self.n_class, (1, 1, 1), padding='same', activation='sigmoid')(x)
                x = Reshape((self.n_class,))(x)
            else:
                x = Conv3D(self.n_class, (1, 1, 1), padding='same', activation='softmax')(x)
                x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)
        return model

if __name__ == "__main__":
    # model = MobileNetV3_Large((16,96,96,3), 1).build()
    model = MobileNetV3_Small((16, 96, 96, 3), 1).build()
    print(model.summary())

