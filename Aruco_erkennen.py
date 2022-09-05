import cv2 as cv
import numpy as np

cap = cv.VideoCapture(3
                      )

aruco_dic = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
aruco_dic2 = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)
aruco_dic3 = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_50)

ploy_punkte = np.zeros((3,2),int)

cont = 0

def line(x,y):
    pass





while True:
    ret ,frame = cap.read()

    #frame = cv.resize(frame,(1020,720))

    fr_gr = cv.cvtColor(frame,cv.COLOR_BGRA2GRAY)

    corner, ids, rej_cor = cv.aruco.detectMarkers(fr_gr,aruco_dic)
    corner2, ids2, rej_cor2 = cv.aruco.detectMarkers(fr_gr,aruco_dic2)
    corner3, ids2, rej_cor2 = cv.aruco.detectMarkers(fr_gr, aruco_dic3)


    print(corner,ids)
    #aruco_frame = cv.aruco.drawDetectedMarkers(image=frame,corners=corner,ids=ids,borderColor=(0,0,255))

    for  idx in corner:

        x, y, h, w = cv.boundingRect(idx)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[0] = x,y
    for  idx2 in corner2:

        x, y, h, w = cv.boundingRect(idx2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[1] = x,y
    for  idx3 in corner3:

        x, y, h, w = cv.boundingRect(idx3)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[2] = x,y




    print(ploy_punkte)
    pol = cv.polylines(frame,[ploy_punkte],False,(0,0,255),5)




    cv.imshow("bild",frame)
    #cv.imshow("bild1", fr_gr)


    if cv.waitKey(1)== ord("q"):
        break
cap.release()
cv.destroyAllWindows()