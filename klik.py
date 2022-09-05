import cv2 as cv
import numpy as np


cap = cv.VideoCapture(2)

var2 = np.zeros((4,2),int)
count = 0
var = np.zeros((2,2),int)
var_roi = np.zeros((2,2),int)
"""var4 = np.empty((4,2),dtype=int)
var3 = np.full((4,2),-1,dtype=int)"""
def kli(event,x,y,flags,params):
    global count


    if event == cv.EVENT_LBUTTONDOWN:

        var2[count] = x,y
        count += 1
        #var = var2[0],var2[3]
        print(x,y)
        print(var_roi)
        print(var_roi[1][1])
        print(var2)
        print(var2[0][0])
        if count == 1:
            var[0] = x,y
            var_roi[0] = x, y


        if count == 3:
            var_roi[1] = x ,y

        if count == 4:
            var[1] = x,y
            #np.asarray(var_roi)

"""        if count == 4:
            count = 0
"""




while True:




    ret, frame = cap.read()



    for idx in range(0, 4):
       cirlcle = cv.circle(frame, (var2[idx]), 5, (0, 0, 255),-1)



    #if count > 2:


    pol = cv.polylines(frame,[var2],False,(255,0,0))

    pol2 = cv.polylines(frame,[var],False,(255,0,0))


    if count == 4:
        roi = frame[var_roi[0][1]:var_roi[1][1],var_roi[0][0]:var_roi[1][0]]
        cv.imshow("ROI",roi)











    cv.imshow("Frame", frame)
    cv.setMouseCallback("Frame", kli)


    if cv.waitKey(1) == ord("q"):
        break

