import warnings
from mobilenet_base import MobileNetV3_Small
warnings.filterwarnings('ignore')

# deep_sort imports
import os
from timeit import time
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

# i3d imports
from i3d_inception import Inception_Inflated3d, twohead_Inception_Inflated3d
import argparse
import cv2
import numpy as np
from data_helper import compute_flow
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.python.util.deprecation as deprecation
import time
import matplotlib.pyplot as plt
from outline import create_outline
from keras_radam import RAdam
from keras_lookahead import Lookahead

# confirm TensorFlow sees the GPU
from tensorflow.python.client import device_lib
assert 'GPU' in str(device_lib.list_local_devices())

# confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
from keras import backend
assert len(backend.tensorflow_backend._get_available_gpus()) > 0

# deep_sort params
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# i3d params
dim = (168, 64)
n_sequence = 64
n_channels = 3 # color channel(RGB)
min_frames = 64 # need 50 frames before predict

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='Modality of images used', required=True)
parser.add_argument('--path', help='Path of model weights', required=True)
parser.add_argument('--video', help='Path of video', required=True)
parser.add_argument('--batch', help='To do batching or not', required=True) # True/False
args = vars(parser.parse_args())

from keras.utils import multi_gpu_model
from keras.models import load_model
# model = Inception_Inflated3d(num_frames = n_sequence, classes=3, input_shape=(32,168,64,3))
# p_model = multi_gpu_model(model, gpus=1)
# p_model.load_weights(args['path'])  # load multi-gpu model weights
# old_model = p_model.layers[-2]   #get single GPU model weights
# # it's necessary to save the model before use this single GPU model
# old_model.save("singlu_gpu.hdf5")
# model.load("single_gpu.hdf5")
# model = load_model(args['path'])
model = load_model(args['path'], compile=False).layers[-2]
model.compile(optimizer = Lookahead(RAdam()), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model = Inception_Inflated3d(num_frames=64, classes=3, input_shape=(64,168,64,1))
# model = model.load_weights(args['path']).layers[-2]

print(args['mode'], "Model loaded")

cap = cv2.VideoCapture(args['video'])
print("Video loaded")

w = int(cap.get(3))
h = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
print("creating writer")
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (w, h))
print("writer created")

yolo = YOLO()
print("YOLO instantiated")

# classes = ['fighting', 'sitting', 'standing', 'walking']
classes = ['not fighting', 'fighting']
# classes = ['fighting', 'standing', 'waling']
base_path = "output"
os.mkdir(base_path)
os.mkdir('output\\fighting')
seen_track_ids = []
# sampling_index =  np.linspace(0, min_frames-1, n_sequence, dtype='int')  # last frame gives error sometimes, skip it
sampling_index = range(0, 64)
print(sampling_index)
fps = 0.0

def adjust_tracks(tracks, previous_tracks, thres=.1):
    if previous_tracks != None and tracks != None:
        for index, curr_track in enumerate(tracks):
            for prev_track in previous_tracks:
                if curr_track.track_id == prev_track.track_id:
                    curr_bbox = curr_track.to_tlbr()
                    prev_bbox = prev_track.to_tlbr()

                    curr_height = float(curr_bbox[2] - curr_bbox[0])
                    curr_width = float(curr_bbox[3] - curr_bbox[1])

                    prev_height = float(prev_bbox[2] - prev_bbox[0])
                    prev_width = float(prev_bbox[3] - prev_bbox[1])

                    delta_height = abs((curr_height-prev_height) / curr_height)
                    delta_width = abs((curr_width-prev_width) / curr_width)
                    print(delta_height, delta_width)

                    if delta_height > thres or delta_width > thres:
                        print('adjusted')
                        tracks[index].mean[2:4] = prev_track.mean[2:4] # force it to the previous aspect ratio and height
    return tracks

def process_track(tracks, frame, previous_tracks):

    tracks_to_predict = []

    # tracks = adjust_tracks(tracks, previous_tracks)

    for track in tracks:
        if track.is_confirmed() and track.time_since_update <= 1:

            bbox = track.to_tlbr()

            temp = frame.copy()
            width = int(bbox[2]) - int(bbox[0])
            height = int(bbox[3]) - int(bbox[1])

            # add code here to retrieve old Bbox, return it as temp
            # need old Bbox
            #
            temp = temp[(int(bbox[1])) - 10:(int(bbox[1])) + height + 10, (int(bbox[0])) - 10:(int(bbox[0])) + width + 10]

            try:
                if track.track_id not in seen_track_ids:
                    os.mkdir(base_path + '\\' + str(track.track_id))  # create the folder for the id first
                    seen_track_ids.append(track.track_id)
                    im = Image.fromarray(temp[..., ::-1])
                    im = im.resize((64,168))
                    im.save(base_path + '\\' + str(track.track_id) + '\\' + str(
                        len(os.listdir(base_path + '\\' + str(track.track_id)))) + '.jpg')
                else:
                    im = Image.fromarray(temp[..., ::-1])
                    im = im.resize((64,168))
                    im.save(base_path + '\\' + str(track.track_id) + '\\' + str(
                        len(os.listdir(base_path + '\\' + str(track.track_id)))) + '.jpg')
            except:
                pass

            if len(os.listdir(base_path + '\\' + str(track.track_id))) >= min_frames:
                tracks_to_predict.append((track.track_id, bbox)) # need their bbox to figure out where to put text

    return tracks_to_predict

THRESHOLD = 0.5

def predict_batch(tracks_to_predict, frame, batch):
    if batch:
        X = np.empty((len(tracks_to_predict), n_sequence, *dim, n_channels))

        for track_num, elem in enumerate(tracks_to_predict):
            track_id = elem[0]
            images = os.listdir(base_path + '\\' + str(track_id))
            images = [int(image.replace('.jpg', '')) for image in images]
            images.sort()
            last_50_images = images[-min_frames:]
            for frame_num, index in enumerate(sampling_index):
                image = cv2.imread(base_path + '\\' + str(track_id) + '\\' + str(last_50_images[index]) + '.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                X[track_num][frame_num] = image/255.0

        # try:
        #     plot_16_frames(X[0])
        # except:
        #     pass

        probs = model.predict(X) # length = len(tracks_to_predict), 17x1 shape
        # pred_classes = [int(round(prob[0] - THRESHOLD + 0.5)) for prob in probs]

        # pred_classes = [process_prob(prob) for prob in probs]
        pred_classes = [np.argmax(prob) for prob in probs]

        for row, pred_class in enumerate(pred_classes):
            bbox = tracks_to_predict[row][1]
            if pred_class == 0:
                # plot_16_frames(X[index])
                id = str(len(os.listdir('output\\fighting')))
                new_fight_path = 'output\\fighting' + '\\' + id
                os.mkdir(new_fight_path)

                for i in range(n_sequence):
                    plt.imsave(new_fight_path + '\\' + id + '_' + str(i) + '.jpg', X[row][i])
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                cv2.putText(frame, id, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 0, 255), 2)

            else:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    else:
        X = np.empty((1, n_sequence, *dim, n_channels))
        for track_num, elem in enumerate(tracks_to_predict):

            track_id = elem[0]
            images = os.listdir(base_path + '\\' + str(track_id))
            images = [int(image.replace('.jpg', '')) for image in images]
            images.sort()
            last_50_images = images[-min_frames:]
            for frame_num, index in enumerate(sampling_index):
                image = cv2.imread(base_path + '\\' + str(track_id) + '\\' + str(last_50_images[index]) + '.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if args['mode'] == 'RGB':
                    X[0][frame_num] = image / 255.0
                elif args['mode'] == "Flow":
                    X[0][frame_num] = image

            if args['mode'] == "Flow":
                tmp = compute_flow(X[0].astype('uint8')) / 255.0
                tmp = np.expand_dims(tmp, 0)

            try:
                probs = model.predict(create_outline((X*255).astype('uint8'))) # RGB
            except:
                probs = model.predict(tmp) # Flow
            pred_class = np.argmax(probs[0])

            bbox = elem[1]  # second element of each tuple
            # cv2.putText(frame, classes[pred_class], (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 0, 255), 2)
            if pred_class == 0:
                # plot_16_frames(X[index])
                id = str(len(os.listdir('output\\fighting')))
                new_fight_path = 'output\\fighting' + '\\' + id
                os.mkdir(new_fight_path)

                for i in range(n_sequence):
                    if args['mode'] == "RGB":
                        plt.imsave(new_fight_path + '\\' + id + '_' + str(i) + '.jpg', X[0][i])
                    elif args['mode'] == "Flow":
                        plt.imsave(new_fight_path + '\\' + id + '_' + str(i) + '.jpg', X[0][i]/255.0)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                cv2.putText(frame, id, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 0, 255), 2)

            else:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    # probs = model.predict(X)
    # pred_classes = [np.argmax(prob) for prob in probs]
    #
    # for index, pred_class in enumerate(pred_classes):
    #
    #     bbox = tracks_to_predict[index][1] # second element of each tuple
    #     # cv2.putText(frame, classes[pred_class], (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 0, 255), 2)
    #     if pred_class == 0:
    #         # plot_16_frames(X[index])
    #         id = str(len(os.listdir('output\\fighting')))
    #         new_fight_path = 'output\\fighting' + '\\' + id
    #         os.mkdir(new_fight_path)
    #
    #         for i in range(n_sequence):
    #             plt.imsave(new_fight_path + '\\' + id + '_' + str(i) + '.jpg', X[index][i])
    #         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 2)
    #         cv2.putText(frame, id,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,0,255),2)
    #
    #     else:
    #         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,0), 2)

def process_prob(prob):
    if np.argmax(prob) == 0 and prob[0] > 0.9:
        return 0
    else:
        prob[0] = 0
        return np.argmax(prob)

def plot_16_frames(arr):
    fig, axs = plt.subplots(4,4)

    for i in range(4):
        axs[i, 0].imshow(arr[i*4])
        axs[i, 1].imshow(arr[i*4 + 1])
        axs[i, 2].imshow(arr[i*4 + 2])
        axs[i, 3].imshow(arr[i*4 + 3])
    plt.show()

def process_det(det, frame):
    bbox = det.to_tlbr()
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)  # blue

previous_tracks = None
while True:
    ret, frame = cap.read()
    if ret != True: # end of video
        break


    image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
    boxs = yolo.detect_image(image)

    features = encoder(frame, boxs)
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections)

    t1 = time.time()

    tracks_to_predict = process_track(tracker.tracks, frame, previous_tracks)
    previous_tracks = tracker.tracks
    if len(tracks_to_predict) != 0:
        predict_batch(tracks_to_predict, frame, False)

    print(time.time()-t1)
    out.write(frame)
    try:
        cv2.imshow('processed', frame)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # fps = (fps + (1. / (time.time() - t1))) / 2
    # print("fps:", fps)
out.release()
cv2.destroyAllWindows()