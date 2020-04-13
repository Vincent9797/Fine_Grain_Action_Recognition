import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability = 15, useHistory = True, maxPixelStability = 15*5, isParallel = True)

while True:
    ret, frame = cap.read()

    # bg subtraction
    fgmask = fgbg.apply(frame)
    fgmask = cv2.resize(fgmask, (500,500))

    frame = cv2.resize(frame, (500,500))
    fin = cv2.bitwise_and(frame, frame, mask=fgmask)
    cv2.imshow("Final image.jpg", fin)

    cv2.imshow('fg', fgmask)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
