#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import warnings
warnings.filterwarnings('ignore')

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


def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    videos = os.listdir('videos')

    for video in videos:
        video_capture = cv2.VideoCapture("videos" + "\\" + video)
        seen_track_ids = []

        if 'mp4' in video:
            base_path = video.split('.mp4')[0]
        elif 'avi' in video:
            base_path = video.split('.avi')[0]
        os.mkdir(base_path)

        while True:
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret != True:
                break

            image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            boxs = yolo.detect_image(image)

            features = encoder(frame,boxs)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                temp = frame.copy()
                width = int(bbox[2]) - int(bbox[0])
                height = int(bbox[3]) - int(bbox[1])

                # 5% border
                temp = temp[(int(bbox[1] - 0.05*height)):(int(bbox[1] + 1.05*height)), (int(bbox[0] - 0.05*width)) - 10:(int(bbox[0] + 1.05*width))]

                try:
                    if track.track_id not in seen_track_ids:
                        os.mkdir(base_path + '\\' + str(track.track_id))  # create the folder for the id first
                        seen_track_ids.append(track.track_id)
                        im = Image.fromarray(temp[..., ::-1])
                        im = im.resize((224,224))
                        im.save(base_path + '\\' + str(track.track_id) + '\\' + str(len(os.listdir(base_path + '\\' + str(track.track_id)))) + '.jpg')
                    else:
                        im = Image.fromarray(temp[..., ::-1])
                        im = im.resize((224,224))
                        im.save(base_path + '\\' + str(track.track_id) + '\\' + str(len(os.listdir(base_path + '\\' + str(track.track_id)))) + '.jpg')
                except:
                    pass

            for det in detections:
                bbox = det.to_tlbr()
                # cv2.rectangle takes in top left and bottom right coordinate
                temp = frame.copy()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2) # blue

            width = int(bbox[2])- int(bbox[0])
            height = int(bbox[3]) - int(bbox[1])
            temp = temp[(int(bbox[1]))-10:(int(bbox[1]))+height+10, (int(bbox[0]))-10:(int(bbox[0]))+width+10]

            # try:
            #     temp = cv2.resize(temp, (int(w/2),h))
            #     cv2.imshow('extracted', temp)
            #     cv2.imshow('', frame)
            # except:
            #     pass

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
