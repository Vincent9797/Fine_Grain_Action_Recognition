import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.layers import Dense, Dropout, Input, InputLayer, Activation, BatchNormalization, Conv2D, Reshape, Concatenate, Lambda, Permute, Multiply, Add, TimeDistributed
from keras.layers import Layer
from keras.models import Model
from keras.applications.resnet50 import ResNet50
import keras.backend as K
import math
import numpy as np


def feedforward(x, d_model, d_ff=2048, dropout=0.3):
    x = Dense(d_ff)(x)
    x = Dropout(dropout)(x)
    x = Dense(d_model)(x)
    return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = Multiply()([q, k])
    scores = Lambda(lambda x: K.sum(x, -1) / math.sqrt(d_k))(scores)

    scores = Activation('softmax')(scores)
    scores = Lambda(lambda x: K.expand_dims(x, axis=-1))(scores)
    scores_size = K.shape(scores)
    v_size = K.shape(v)

    scores = Lambda(lambda x: K.repeat_elements(x, K.get_value(v_size[-1]), 2))(scores)
    scores = Lambda(
        lambda x: K.reshape(x, (K.get_value(scores_size[0]), K.get_value(scores_size[1]), K.get_value(v_size[-1]))),
        output_shape=(K.get_value(scores_size[1]), K.get_value(v_size[-1])))(scores)
    output = Multiply()([scores, v])
    output = Lambda(lambda x: K.sum(x, 1))(output)
    if dropout:
        output = dropout(output)
    return output


class Norm(Layer):
    def __init__(self, d_model, eps=1e-6, trainable=True, **kwargs):
        self.size = d_model
        self.eps = eps
        self.trainable = trainable
        super(Norm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha',
                                     shape=(self.size,),
                                     initializer='zeros',
                                     trainable=self.trainable)
        self.bias = self.add_weight(name='beta',
                                    shape=(self.size,),
                                    initializer='zeros',
                                    trainable=self.trainable)
        super(Norm, self).build(input_shape)

    def call(self, x):
        mu = K.mean(x, axis=-1, keepdims=True)
        mu = K.repeat_elements(mu, self.size, axis=-1)

        sigma = K.std(x, axis=-1, keepdims=True)
        sigma = K.repeat_elements(sigma, self.size, axis=-1)

        norm = self.alpha * (x - mu) / (sigma + self.eps) + self.bias
        return norm

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.size)


class PositionalEncoder(Layer):
    def __init__(self, d_model, max_seq_len=80, **kwargs):
        self.d_model = d_model

        pe = np.zeros(shape=(max_seq_len, d_model))

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = K.constant(pe)
        self.pe = K.expand_dims(pe, 0)
        super(PositionalEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PositionalEncoder, self).build(input_shape)

    def call(self, x):
        x = x * math.sqrt(self.d_model)
        shape = x.shape
        seq_len = shape[1]

        z = K.variable(value=self.pe[:, :seq_len])
        z = K.expand_dims(z, axis=-1)
        z = K.expand_dims(z, axis=-1)

        z = K.repeat_elements(z, shape[-2], -2)
        z = K.repeat_elements(z, shape[-1], -1)
        z = K.repeat_elements(z, shape[0], 0)
        x = x + z
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def TX(q, k, v, mask=None, d_model=64, dropout=0.3):
    dropout_1 = Dropout(dropout)
    dropout_2 = Dropout(dropout)
    norm_1 = Norm(d_model)
    norm_2 = Norm(d_model)

    b = K.shape(q)[0]
    t = K.shape(k)[1]
    dim = K.shape(q)[1]

    q_temp = Lambda(lambda x: K.expand_dims(x, axis=1))(q)
    q_temp = Lambda(lambda x: K.repeat_elements(q_temp, K.get_value(t), 1))(q_temp)
    q_temp = Lambda(lambda x: K.reshape(q_temp, (b, t, dim)))(q_temp)

    A = attention(q_temp, k, v, d_model, mask, dropout_1)
    q_ = Add()([A, q])
    q_ = norm_1(q_)

    q_ = Add()([q_, dropout_2(feedforward(q_, d_model, d_model // 2))])
    new_query = norm_2(q_)
    return new_query


def Block_head(q, k, v, mask=None, d_model=64, dropout=0.3):
    q = TX(q, k, v)
    q = TX(q, k, v)
    q = TX(q, k, v)
    return q


def Tail(x, num_classes, num_frames, b, t, head=16):
    spatial_h = 6
    spatial_w = 2
    num_features = 2048
    d_model = num_features // 2
    d_k = d_model // head

    # b = x.shape[0] // t
    # print("b in Tail:", b)
    x = BatchNormalization()(x)
    x = Permute((3, 1, 2), input_shape=(spatial_h, spatial_w, num_features))(x) # put channel first
    x = Lambda(lambda x: K.reshape(x, (b, t, num_features, spatial_h, spatial_w)))(x)
    x = PositionalEncoder(num_features, num_frames)(x)
    x = Lambda(lambda x: K.reshape(x, (b * t, num_features, spatial_h, spatial_w)))(x)
    x = Conv2D(d_model, kernel_size=(spatial_h, spatial_w), strides=1, padding='valid', bias=False,
               data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = Lambda(lambda x: K.reshape(x, (b, t, d_model)))(x)
    x = Norm(d_model, trainable=False)(x)

    q = Lambda(lambda x: x[:, t // 2, :], output_shape=(b, d_model))(x)
    v = x
    k = x

    q = Lambda(lambda _x: K.reshape(_x, (b, head, d_k)), output_shape=(head, d_k))(q)
    k = Lambda(lambda _x: K.reshape(_x, (b, t, head, d_k)), output_shape=(t, head, d_k))(k)
    v = Lambda(lambda _x: K.reshape(_x, (b, t, head, d_k)), output_shape=(t, head, d_k))(v)
    k = Permute((2, 1, 3), input_shape=(t, head, d_k))(k)
    v = Permute((2, 1, 3), input_shape=(t, head, d_k))(v)

    outputs = []
    for i in range(head):
        print("Attention head", i)
        q_input = Lambda(lambda x: x[:, i], output_shape=(t,))(q)
        k_input = Lambda(lambda x: x[:, i], output_shape=(t, d_k))(k)
        v_input = Lambda(lambda x: x[:, i], output_shape=(t, d_k))(v)
        tmp = Block_head(q_input, k_input, v_input)
        outputs.append(tmp)

    f = Concatenate(axis=1)(outputs)
    f = Lambda(lambda x: K.l2_normalize(x, axis=None))(f)
    y = Dense(num_classes, activation="softmax")(f)
    return y


def vatn(b, t):
    input_ = Input(batch_shape=(b, t, 168, 64, 3))
    # print("FIRST INPUT", input_.shape)
    x = Lambda(lambda x: K.reshape(x, (b*t, 168, 64, 3)))(input_)
    # x = TimeDistributed(ResNet50(weights='imagenet', include_top=False))(input_)
    x = ResNet50(weights='imagenet', include_top=False)(x)
    x = Tail(x, num_classes=3, num_frames=t, b=b, t=t)
    model = Model(input_, x)

    return model

# model = VATN(2, 64)
# print(model.summary())