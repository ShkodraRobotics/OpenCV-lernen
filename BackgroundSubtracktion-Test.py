import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

mog = cv.createBackgroundSubtractorMOG2()



while True:
    ret, frame = cap.read()

    back = mog.apply(frame)

    cv.imshow("IMG",back)


    if cv.waitKey(1) == ord("q"):
      break


cap.release()
cv.destroyAllWindows()







