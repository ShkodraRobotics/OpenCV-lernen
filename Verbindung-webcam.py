from matplotlib import pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,25,28)
    conturen = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    cv2.imshow("frame", conturen)

    if cv2.waitKey(1)== ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

