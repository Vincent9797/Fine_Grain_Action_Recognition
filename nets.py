import tensorflow as tf
import slowfast
from keras.layers import Input
from slowfast import TimeIndexing

__all__=['network']

def resnet50(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 4, 6, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet101(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 4, 23, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet152(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 8, 36, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet200(inputs, **kwargs):
    model = slowfast.Slow_body(inputs, [3, 24, 36, 3], slowfast.bottleneck, **kwargs)
    return model



network = {
    'resnet50':resnet50,
    'resnet101':resnet101,
    'resnet152':resnet152,
    'resnet200':resnet200
}

import numpy as np
from keras.utils import multi_gpu_model
from keras.models import load_model
if __name__=="__main__":
    inputs = Input(shape=(64, 168, 64, 3))
    model = resnet50(inputs, num_classes=3)
    # model = multi_gpu_model(model, 2)
    # print(model.summary())

    # model.base_model.save('my_model.h5')
    # # model.save_weights('my_model_weights.h5')
    # del model
    # model = load_model('my_model.h5')
    # model = resnet50(inputs, num_classes=3)
    # model = multi_gpu_model(model, 2)
    model.load_weights('save_weight/weight-01-0.82-0.63-1.70101.hdf5')
    print(model.summary())
    



