import cv2
import numpy as np

def readfile_to_dict(filename):
    'Read text file and return it as dictionary'
    d = {}
    f = open(filename)
    for line in f:
        # print(str(line))
        if line != '\n':
            (key, val) = line.split()
            d[key] = int(val)

    return d

def calculateRGBdiff(sequence_img):
    'keep first frame as rgb data, other is use RGBdiff for temporal data'
    length = len(sequence_img)        
    # find RGBdiff frame 2nd to last frame
    for i in range(length-1,0,-1): # count down
        sequence_img[i] = cv2.subtract(sequence_img[i],sequence_img[i-1])
    return sequence_img

def compute_flow(sequence_img, bound=15): # sequence image of shape (num_frames, height, width, 3), returns (num_frames-1, height, width, 3)
    hsv = np.zeros_like(sequence_img[0])
    hsv[..., 1] = 255

    for i in range(len(sequence_img)-1):
        prev = sequence_img[i]
        curr = sequence_img[i+1]


        prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        curr = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)


        flow = cv2.calcOpticalFlowFarneback(prev,curr,float(0),float(0),3,15,3,5,float(1),0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2

        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = np.clip(rgb, -20, 20)

        rgb = (rgb+bound) * (255.0/(2*bound))
        rgb = np.round(rgb).astype(int)

        # clip values
        rgb[rgb>=255] = 255
        rgb[rgb<=0] = 0

        sequence_img[i] = rgb
        # sequence_img[i] = rgb/255
    return sequence_img[:len(sequence_img)-1]


def compute_TVL1(prev, curr, bound=15):
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    flow = np.clip(flow, -20,20) #default values are +20 and -20\n",
    assert flow.dtype == np.float32
    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow

def compute_i3d_flow(sequence_img):
    length = len(sequence_img)
    res = np.empty((length-1, sequence_img.shape[1], sequence_img.shape[2], sequence_img.shape[3]-1))
    for i in range(length-1):
        prev = cv2.cvtColor(sequence_img[i], cv2.COLOR_RGB2GRAY)
        curr = cv2.cvtColor(sequence_img[i+1], cv2.COLOR_RGB2GRAY)
        res[i] = compute_TVL1(prev, curr)
    return res