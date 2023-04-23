import cv2 as cv
import numpy as np


cap = cv.VideoCapture(2)
img = cv.imread("d.jpg")
img = cv.resize(img,(500,500))
img_gr = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
rot,im_th = cv.threshold(img_gr,80,255,cv.THRESH_BINARY_INV)

def no(x):
    pass

cv.namedWindow("Bild")
cv.createTrackbar("thresh","Bild",0,255,no)
while True:
    ret, frame = cap.read()



    T = cv.getTrackbarPos("thresh","Bild")

    fr_gr = cv.cvtColor(frame,cv.COLOR_BGRA2GRAY)
    rot, fr_th = cv.threshold(fr_gr,T,255,cv.ADAPTIVE_THRESH_MEAN_C)



    con, hir = cv.findContours(im_th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    con_F, hir_F = cv.findContours(fr_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    conm = cv.drawContours(img,con,-1,(0,0,255),3)

    for idx in con_F:
        M = cv.moments(idx)
        A = cv.contourArea(idx)



        ret_1 = cv.matchShapes(con[0],idx, cv.CONTOURS_MATCH_I2,0.0)

        print(ret_1)

        if ret_1 <1.2 and  A > 100:
            x ,y ,w, h = cv.boundingRect(idx)
            x_c = x + (w / 2)
            y_c = y + (h / 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)






    cv.imshow("Bild",fr_th)
    cv.imshow("Bild2", im_th)
    cv.imshow("Bild3", frame)
    #cv.imshow("Bild4", conm)

    if cv.waitKey(1) == ord("q"):
        break
