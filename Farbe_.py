import cv2 as cv
import numpy as np

cap= cv.VideoCapture(0)

Blau_low = np.array([60,35,140])
Blau_hi = np.array([180,255,255])



while True:
    ret, Frame = cap.read()

    hsv = cv.cvtColor(Frame,cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, Blau_low, Blau_hi)



    result = cv.bitwise_and(Frame, Frame, mask=mask)



    cv.imshow("Cam",result)




    if cv.waitKey(1) == ord("q"):
        break
cap.release()
cv.destroyAllWindows()









