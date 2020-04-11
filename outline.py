import cv2
import numpy as np

def create_outline(batch_of_frames):

    b = batch_of_frames.shape[0]
    t = batch_of_frames.shape[1]
    res = np.empty((b,t,168,64,1))
    for i in range(b):
        for j in range(t):
            frame = batch_of_frames[i][j]
            img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

            img_threshold1 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 3)

            img_threshold1_blurred = cv2.GaussianBlur(img_threshold1, (5, 5), 0)

            _, img_threshold2 = cv2.threshold(img_threshold1_blurred, 200, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
            img_opening = cv2.bitwise_not(cv2.morphologyEx(cv2.bitwise_not(img_threshold2), cv2.MORPH_OPEN, kernel))

            img_opening_blurred = cv2.GaussianBlur(img_opening, (3, 3), 0)

            img_opening_blurred = np.expand_dims(img_opening_blurred, -1)

            res[i][j] = img_opening_blurred

    return res