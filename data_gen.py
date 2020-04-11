import numpy as np
import keras
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_helper import calculateRGBdiff, compute_flow, compute_i3d_flow
from albumentations.augmentations.functional import brightness_contrast_adjust
import albumentations as A
import random
import keras.backend as K
from outline import create_outline

class IndependentRandomBrightnessContrast(A.ImageOnlyTransform):
    """ Change brightness & contrast independently per channels """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5):
        super(IndependentRandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = A.to_tuple(brightness_limit)
        self.contrast_limit = A.to_tuple(contrast_limit)

    def apply(self, img, **params):
        img = img.copy()
        for ch in range(img.shape[2]):
            alpha = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])
            beta = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
            img[..., ch] = brightness_contrast_adjust(img[..., ch], alpha, beta)

        return img

albu_tfms =  A.Compose([
    A.OneOf([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, value=0),
        A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0),
        A.NoOp()
    ]),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.5,
                                   contrast_limit=0.4),
        IndependentRandomBrightnessContrast(brightness_limit=0.25,
                                                        contrast_limit=0.24),
        A.RandomGamma(gamma_limit=(50, 150)),
        A.NoOp()
    ]),
    A.OneOf([
        A.FancyPCA(),
        A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
        A.HueSaturationValue(hue_shift_limit=5,
                             sat_shift_limit=5),
        A.NoOp()
    ]),
    A.OneOf([
        A.CLAHE(),
        A.NoOp()
    ]),
    A.HorizontalFlip(p=0.5),
])

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32), 
                n_channels=1,n_sequence=4, shuffle=True, path_dataset=None,
                type_gen='train', option=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_sequence = n_sequence
        self.shuffle = shuffle
        self.path_dataset = path_dataset
        self.type_gen = type_gen
        self.option = option
        self.aug_gen = ImageDataGenerator() 
        print("all:", len(self.list_IDs), " batch per epoch", int(np.floor(len(self.list_IDs) / self.batch_size)) )
        self.on_epoch_end()

        self.n = 0
        self.max = self.__len__()

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        if self.type_gen == 'predict':
            return X
        else:
            return X, y

    def get_sampling_frame(self, len_frames):
        return range(self.n_sequence)
        # return np.linspace(1, len_frames-2, self.n_sequence, dtype='int') # last frame gives error sometimes, skip it

    def sequence_augment(self, sequence):
        name_list = ['rotate','width_shift','height_shift',
                    'brightness','flip_horizontal','width_zoom',
                    'height_zoom']
        dictkey_list = ['theta','ty','tx',
                    'brightness','flip_horizontal','zy',
                    'zx']

        random_aug = np.random.randint(2, 5) # random 2-4 augmentation method
        pick_idx = np.random.choice(len(dictkey_list), random_aug, replace=False) #

        dict_input = {}
        for i in pick_idx:
            if dictkey_list[i] == 'theta':
                dict_input['theta'] = np.random.randint(-10, 10)

            elif dictkey_list[i] == 'ty': # width_shift
                dict_input['ty'] = np.random.randint(-10, 10)

            elif dictkey_list[i] == 'tx': # height_shift
                dict_input['tx'] = np.random.randint(-10, 10)

            elif dictkey_list[i] == 'brightness': 
                dict_input['brightness'] = np.random.uniform(0.8,1.2)

            elif dictkey_list[i] == 'flip_horizontal': 
                dict_input['flip_horizontal'] = True

            elif dictkey_list[i] == 'zy': # width_zoom
                dict_input['zy'] = np.random.uniform(0.75,1.25)

            elif dictkey_list[i] == 'zx': # height_zoom
                dict_input['zx'] = np.random.uniform(0.75,1.25)
        len_seq = sequence.shape[0]
        for i in range(len_seq):
            sequence[i] = self.aug_gen.apply_transform(sequence[i],dict_input)
        
        return sequence

    def albu_aug(self, sequence, tfms = albu_tfms):
        seed = random.randint(0, 99999)


        for index, x in enumerate(sequence):
            random.seed(seed)
            sequence[index] = tfms(image=x.astype('uint8'))['image']
        return sequence

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization

        if self.option == "Fuse":
            X = np.empty((self.batch_size, self.n_sequence*2-1, *self.dim, self.n_channels)) # (bs, 127, 224, 224, 3)
        elif self.option == "Flow":
            X = np.empty((self.batch_size, self.n_sequence-1, *self.dim, self.n_channels)) # (bs, 63, 224, 224, 3)
        else:
            X = np.empty((self.batch_size, self.n_sequence, *self.dim, self.n_channels)) # (bs, 64, 224, 224, 3)
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):  # ID is name of file
            path_file = self.path_dataset + ID + '.avi'
            cap = cv2.VideoCapture(path_file)
            length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get how many frames this video have
            index_sampling = self.get_sampling_frame(length_file) # get sampling index, returns 64 index

            if self.option == "Flow":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            for j, n_pic in enumerate(range(self.n_sequence)):
                # cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic) # jump to that index
                ret, frame = cap.read()

                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # forgot to do it
                    frame = cv2.resize(frame, self.dim[::-1])
                except:
                    print(frame, length_file, path_file, "Using old image")
                X[i,j,:,:,:] = frame

            if self.option != 'RGB': # for fuse and flow
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # forgot to do it
                frame = cv2.resize(frame, self.dim[::-1])

                temp = (X[i,]).astype('uint8')
                temp = np.insert(temp, 0, frame, axis=0)
                if self.option == 'Flow':
                    X[i,] = compute_flow(temp)
                elif self.option == 'Fuse': # using both RGB + Flow => X will have shape of (BS, 2*n_sequnece, 224, 224, 3)
                    X[i,] = np.concatenate((X[i,:self.n_sequence], compute_flow(temp[:self.n_sequence])), axis=0)

            # augmentation and retrieving labels
            if self.type_gen =='train':
                X[i,] = (self.sequence_augment(X[i,]))/255.0 # each sample undergoes the same transformation
            else:
                X[i,] = X[i,]/255.0
            Y[i] = self.labels[ID]
            cap.release()

        if self.option == "Fuse":
            return [X[:,:self.n_sequence,:,:,:], X[:,self.n_sequence:,:,:,:]], Y
        else:
            if self.option == "Grey":
                X = X*255
                X = X.astype('uint8')
                X = create_outline(X)
            return X, Y

