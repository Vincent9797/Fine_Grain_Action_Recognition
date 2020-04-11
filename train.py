import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

from data_gen import DataGenerator
from data_helper import readfile_to_dict

from sklearn.metrics import f1_score, precision_score, recall_score
import keras
from keras.utils import multi_gpu_model
from keras.models import load_model
import keras.backend as K
from keras.callbacks.callbacks import Callback, ReduceLROnPlateau
from tensorflow.python.keras.callbacks import ModelCheckpoint
from i3d_inception import Inception_Inflated3d, twohead_Inception_Inflated3d, get_model
from keras.layers import Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
from custom_callbacks import F1Callback
from random import random
from collections import Counter
import os

from nets import resnet101, resnet50
from mobilenet_base import MobileNetV3_Large, MobileNetV3_Small
# from shufflenetv2 import ShuffleNetV2
# from effnet import Effnet
# from iotnet import iotnet3
# from VATN import vatn
# import wandb
# wandb.init()
# print("Initialised GPU monitoring")

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', help='Batch size', required=True) # 4 for 2-stream, 8 for 1-stream
parser.add_argument('--mode', help='Modality of images used', required=True) # Either RGB/Flow/Fuse
# parser.add_argument('--dim', help='Dimension of images used', required=True)
parser.add_argument('--num_class', help='Number of classes', required=True)
parser.add_argument('--qs', help='Max queue size', required=True)
parser.add_argument('--workers', help='Num workers', required=True)
parser.add_argument('--gpu', help='Which GPU to use', required=False)
args = vars(parser.parse_args())

if args['gpu'] != None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']

print("----- Using GPU", args['gpu'], "-----")

###### Parameters setting
dim = (168, 64)
n_sequence = 64 # at 64 timeframes
n_channels = 3 # color channel(RGB)
n_output = int(args['num_class']) # number of output class
batch_size = int(args['batchsize'])
n_mul_train = 1 # To increase sample of train set
n_mul_val = 1 # To increase sample of test set
path_dataset = ''
######

# Keyword argument
params = {'dim': dim,
          'batch_size': batch_size, # you can increase for faster training
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
          'option': args['mode'],
          'shuffle': True}

train_txt = "dataset_list/train_64frames.txt"
val_txt = "dataset_list/val_64frames.txt"
# test_txt = "dataset_list/testlist.txt"
train_d = readfile_to_dict(train_txt)
val_d = readfile_to_dict(val_txt)
# test_d = readfile_to_dict(test_txt)


# def select_fights(path):
#
#     elems = path.split('\\')
#     if elems[1] == 'fighting':
#         return True
#     else:
#         return False
# print(len(train_d.keys()))
# train_d = dict(filter(lambda elem: select_fights(elem[0]), train_d.items()))
print(len(train_d.keys()))

print(Counter(list(train_d.values())))
print(Counter(list(val_d.values())))
# print(Counter(list(test_d.values())))

train_keys = list(train_d.keys()) * n_mul_train
val_keys = list(val_d.keys()) * n_mul_val
# test_keys = list(test_d.keys())

# Generators
print("Params for DataGen:", params)
training_generator = DataGenerator(train_keys, train_d, **params, type_gen='train')
validation_generator = DataGenerator(val_keys, val_d, **params, type_gen='test')

# params['shuffle'] = False
# params['batch_size'] = 2
# testing_generator = DataGenerator(test_keys, test_d, **params, type_gen='test')

X, Y = training_generator[0]  # returns variables and labels pair
print(X[0].shape, X[1].shape, Y.shape)
X0 = X[0]
X1 = X[1]
fig = plt.figure(figsize=(8, 8))
columns = 16 if (args['mode'] == 'Fuse') else 8
rows = 8
if args['mode'] == 'Fuse':
    for i in range(1, columns * rows):
        if i > 64:
            fig.add_subplot(rows, columns, i)
            plt.imshow(X1[0][i - 1-64])
        else:
            fig.add_subplot(rows, columns, i)
            plt.imshow(X0[0][i - 1])
    plt.show()
else:
    for i in range(1, columns * rows+1):
        fig.add_subplot(rows, columns, i)
        if args['mode'] == 'Grey':
            plt.imshow(X0[i - 1][:,:,0], cmap='gray')
        else:
            plt.imshow(X0[i - 1])
    plt.show()

X, Y = validation_generator[0]  # returns variables and labels pair
X0 = X[0]
X1 = X[1]
fig = plt.figure(figsize=(8, 8))
columns = 16 if (args['mode'] == 'Fuse') else 8
rows = 8

if args['mode'] == 'Fuse':
    for i in range(1, columns * rows):
        if i > 64:
            fig.add_subplot(rows, columns, i)
            plt.imshow(X1[0][i - 1-64])
        else:
            fig.add_subplot(rows, columns, i)
            plt.imshow(X0[0][i - 1])
    plt.show()
else:
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if args['mode'] == 'Grey':
            plt.imshow(X0[i - 1][:, :, 0], cmap='gray')
        else:
            plt.imshow(X0[i - 1])
    plt.show()

# if args['mode'] == 'Grey':
#     model = get_model(args['mode'], n_sequence, n_output, dim, channels=1)
# else:
#     model = get_model(args['mode'], n_sequence, n_output, dim, channels=3)
single_model = resnet50(Input(shape=(64, 168, 64, 3)), num_classes=3)
# model = MobileNetV3_Large((n_sequence, dim[0], dim[1], n_channels), n_output).build()
# model = MobileNetV3_Small((n_sequence, dim[0], dim[1], n_channels), n_output).build()
# model = ShuffleNetV2(input_shape = (n_sequence, dim[0], dim[1], n_channels), classes=n_output)
# model = Effnet((n_sequence, dim[0], dim[1], n_channels), n_output)
# model = iotnet3((16,96,96,3), 1, n=4, k=0.7)

# Load weight of unfinish training model(optional)
load_model = True
start_epoch = 0
if load_model:
    weights_path = 'save_weight/weight-41-0.98-0.64-0.45690.hdf5' # name of model
    start_epoch = 41
    single_model.load_weights(weights_path)
    # model = load_model(weights_path)

model = multi_gpu_model(single_model, gpus=2)
# model = load_model('save_weight/weight-03-0.93-0.65-0.11879.hdf5')
print(model.summary())

def categorical_focal_loss(gamma=2., alpha=.25):
    def categorical_focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return categorical_focal_loss_fixed

# to not change data_gen
def focal_loss(gamma=2., alpha=.25):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_true = keras.utils.to_categorical(y_true, num_classes=3)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return categorical_focal_loss_fixed

from keras_radam import RAdam
from keras_lookahead import Lookahead

optimizer = Lookahead(RAdam())
if n_output == 1:
    print("Binary")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
else:
    print("Multi-class")
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=0.0008), loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=['accuracy'])

# from keras.utils import plot_model
# plot_model(model, "se_inception.png", show_shapes=True)



# Set callback
validate_freq = 1
filepath = "save_weight/"+"weight-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}-{val_loss:.5f}.hdf5"

# save single model
class MyModelCheckPoint(ModelCheckpoint):

    def __init__(self, singlemodel, *args, **kwargs):
        self.singlemodel = singlemodel
        super(MyModelCheckPoint, self).__init__(*args, **kwargs)
    def on_epoch_end(self, epoch, logs=None):
        self.model = self.singlemodel
        super(MyModelCheckPoint, self).on_epoch_end(epoch, logs)


# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, period=validate_freq)
checkpoint = MyModelCheckPoint(single_model, filepath, monitor='val_accuracy', verbose=1, save_best_only=False, period=validate_freq)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.95, patience=2, verbose=1, mode='auto')
# callbacks_list = [F1Callback(testing_generator, 2, validate_freq), # set to highest while still predicting everything
#                   checkpoint, reduce_lr]

callbacks_list = [checkpoint, reduce_lr]

# Train model on dataset
print("FITTING")
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=90,
                    verbose=1,
                    callbacks=callbacks_list,
                    initial_epoch=start_epoch,
                    validation_freq=validate_freq,
                    max_queue_size=int(args['qs']),
                    workers = int(args['workers']),
                    #use_multiprocessing=True
                    )

# model.load_weights("save_weight/weight-51-1.00-0.62.hdf5")
# probs = model.predict_generator(validation_generator)
# print(probs)
# y_pred = [np.argmax(prob) for prob in probs]
# y_true = list(val_d.values())
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_true, y_pred))
# print(accuracy_score(y_true, y_pred))
# print(classification_report(y_true, y_pred))
