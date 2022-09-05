import cv2 as cv
import numpy as np

cap = cv.VideoCapture(2)

kernel = np.ones((5,5),np.uint8)



while (True):
    ret, Frame = cap.read( )
    cap_gery = cv.cvtColor(Frame, cv.COLOR_BGRA2GRAY)
    _, mask = cv.threshold(cap_gery,120,255,cv.THRESH_TOZERO + cv.THRESH_OTSU)
    ad_TH = cv.adaptiveThreshold(cap_gery,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,199,5)

    img_ero = (cv.erode(mask,kernel,iterations=1))

    cv.imshow("Test", img_ero)
    cv.imshow("otsu",mask)
    cv.imshow("test", cap_gery)äölkjhuzrtfdyxa<SRTF%GUZH&/PÜ?
    ´?=)("!"^Qa<vbnjmpöüä+#´ß0)
    cv.imshow("adaptiv", Frame,)
    if cv.waitKey(1) == ord("q"):
        break
cap.release()
cv.destroyAllWindows()

