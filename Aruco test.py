import cv2 as cv
import numpy as np

cap  = cv.VideoCapture(0)

poly_points = np.zeros((2,2),int)


aruco_dic = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)



while True:
    ret ,frame = cap.read()

    #frame = cv.resize(frame,(1020,720))

    fr_gr = cv.cvtColor(frame,cv.COLOR_BGRA2GRAY)


    corner, ids, _ = cv.aruco.detectMarkers(fr_gr,aruco_dic,ids=2)

    lis =[ids,corner]
    #print(lis)
    try:
        print("1"f"{lis[0][0]}")
        print("2"f"{lis[0][1]}")
        print("3"f"{lis[1][0]}")
        print("4"f"{lis[1][1]}")
    except:pass
    try:
        if lis[0][1] == 2:
            cnt = lis[1][1]

            x, y, h, w = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            poly_points[0] = x+(w/2),y+(h/2)

        elif lis[0][0] == 2:
            cnt = lis[1][0]
            x, y, h, w = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            poly_points[0] == x+(w/2),y+(h/2)
        else:pass
    except:pass



    try:
        if lis[0][1] == 0:
            cnt = lis[1][1]

            x, y, h, w = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            poly_points[1] = x + (w / 2), y + (h / 2)

        elif lis[0][0] == 0:
            cnt = lis[1][0]
            x, y, h, w = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            poly_points[1] = x + (w / 2), y + (h / 2)
        else:
            pass
    except:
        pass

    try: pol = cv.polylines(frame, [poly_points], False, (0, 0, 255), 1)
    except:pass
    """for cnt in lis:
        x, y, h, w = cv.boundingRect(cnt)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)"""


    #print(corner)




    #cv.imshow("win", fr_gr)
    cv.imshow("bild1", frame)

    if cv.waitKey(1) == ord("q"):
        break
cap.release()
cv.destroyAllWindows()