import cv2
import numpy as np
from deeplabv3 import Deeplabv3

deeplab_model = Deeplabv3()
vid = cv2.VideoCapture(0)
blurValue = (41,41)
while True:
    ret, frame = vid.read()

    w, h, _ = frame.shape
    ratio = 512. / np.max([w,h])
       
    resized = cv2.resize(frame,(int(ratio*h),int(ratio*w)))
    resized = resized / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')    
    res = deeplab_model.predict(np.expand_dims(resized2,0))
    labels = np.argmax(res.squeeze(),-1)    
    
    labels = labels[:-pad_x-25]
    mask = labels == 0    
    resizedFrame = cv2.resize(frame, (labels.shape[1],labels.shape[0]))
    blur = cv2.GaussianBlur(resizedFrame,blurValue,0)
    cv2.imshow("resized", resizedFrame)


    resizedFrame[mask] = 0
    cv2.imshow("result",resizedFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.stop()
cv2.destroyAllWindows()
