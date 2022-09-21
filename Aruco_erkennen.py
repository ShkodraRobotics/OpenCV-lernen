import cv2 as cv
import numpy as np
import math
from scipy.spatial import distance as dist

cap = cv.VideoCapture(0)



def definer(entscheider_var):
    ent1 = entscheider_var



def no(x):
    pass
aruco_dic = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
aruco_dic2 = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)
aruco_dic3 = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_50)
aruco_dic4 = cv.aruco.Dictionary_get(cv.aruco.DICT_7X7_50)

ploy_punkte = np.zeros((3,2),int)
ploy_punkte2 = np.zeros((2,2),int)
ploy_punkte3 = np.zeros((2,2),int)
ploy_punkte4 = np.zeros((2,2),int)
cont = 0

def line(x,y):
    pass
cv.namedWindow("win")
cv.createTrackbar("t_bar","win",0,255,no)




while True:
    ret ,frame = cap.read()

    #frame = cv.resize(frame,(1020,720))

    fr_gr = cv.cvtColor(frame,cv.COLOR_BGRA2GRAY)

    t = cv.getTrackbarPos("t_bar","win")
    wid = frame.shape[1]


    ok ,th_img = cv.threshold(fr_gr,241,255,cv.ADAPTIVE_THRESH_MEAN_C)
    th_img = np.array(th_img)

    corner, ids, rej_cor = cv.aruco.detectMarkers(fr_gr,aruco_dic)
    corner2, ids2, rej_cor2 = cv.aruco.detectMarkers(fr_gr,aruco_dic2)
    corner3, ids3, rej_cor3 = cv.aruco.detectMarkers(fr_gr, aruco_dic3)
    corner4, ids4, rej_cor4 = cv.aruco.detectMarkers(fr_gr, aruco_dic4)

    #print(corner,ids)
    #aruco_frame = cv.aruco.drawDetectedMarkers(image=frame,corners=corner,ids=ids,borderColor=(0,0,255))
    h,w = 0 , 0

    for  idx in corner:

        x, y, h, w = cv.boundingRect(idx)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[0] = x+(w/2),y+(h/2)
    for  idx2 in corner2:

        x, y, h, w = cv.boundingRect(idx2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[1] = x+(w/2),y+(h/2)
    for  idx3 in corner3:

        x, y, h, w = cv.boundingRect(idx3)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[2] = x+(w/2),y+(h/2)
    for  idx4 in corner4:

        min_aruco = cv.minAreaRect(idx4)
        min_point =cv.boxPoints(min_aruco)
        box = np.int0(min_point)
        cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
        min_punkt_1 = box[0]
        min_punkt_2 = box[1]
        min_punkt_3 = box[2]
        min_punkt_4 = box[3]
        min_punkt_1_x = box[0][0]
        min_punkt_1_y = box[0][1]
        min_punkt_2_x = box[1][0]
        min_punkt_2_y = box[1][1]
        min_punte_distanz = math.sqrt(((min_punkt_1_x - min_punkt_2_x) ** 2) + ((min_punkt_1_y - min_punkt_2_y) ** 2))
        min_punte_distanz_cm = min_punte_distanz / 4.4
        #print(min_punte_distanz)

        cir_1= cv.circle(frame,min_punkt_1,2,color=(0,0,255),thickness=10,lineType=3)
        cir_2 = cv.circle(frame, min_punkt_2, 2, color=(0, 255, 0), thickness=10, lineType=3)
        cir_3 = cv.circle(frame, min_punkt_3, 2, color=(255, 0, 0), thickness=10, lineType=3)
        cir_4 = cv.circle(frame, min_punkt_4, 2, color=(255, 0, 255), thickness=10, lineType=3)

        #print(box)

        """x, y, h, w = cv.boundingRect(idx4)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[2] = x+(w/2),y+(h/2)"""



    ploy_punkte2[0] = ploy_punkte[0]
    ploy_punkte2[1] = ploy_punkte[2]
    #print(w)
    p1x = ploy_punkte2[0][0]
    p1y = ploy_punkte2[0][1]
    p2x = ploy_punkte2[1][0]
    p2y = ploy_punkte2[1][1]
    p3 = math.sqrt(((p1x - p2x)**2) +((p1y - p2y)**2))
    pol = cv.polylines(frame,[ploy_punkte],False,(0,0,255),5)
    pol2 = cv.polylines(frame,[ploy_punkte2],False,(0,255,0))
    ref = w

    try:

        p_dif = p3 / w
        p_dif = p_dif - 2
        #print(round(p_dif))

    except: pass
    try:
        text = cv.putText(frame,f"{p_dif}",(0,100),cv.FONT_HERSHEY_PLAIN,4,(255,0,0),2 )
    except:pass

    th_corn, hi = cv.findContours(th_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)





    for con in th_corn:
        area = cv.contourArea(con)
        if area > 1000:
            per = cv.arcLength(con, True)
            aprox = cv.approxPolyDP(con, 0.03 * per, True)


            if len(aprox) == 4:
                x, y, w, h = cv.boundingRect(con)
                min_r = cv.minAreaRect(con)
                min_x_1 = min_r[0][0]
                min_y_1 = min_r[0][1]
                min_h_ = min_r[1][1]
                p_m_1_x = min_x_1
                p_m_1_y = min_y_1+min_h_ / 2
                p_m_2_x = min_x_1
                p_m_2_y = min_y_1 - min_h_ / 2
                ploy_punkte4[0][0]=min_x_1
                ploy_punkte4[0][1] =p_m_2_y
                ploy_punkte4[1][0] =min_x_1
                ploy_punkte4[1][1] =p_m_1_y
                cv.polylines(frame, [ploy_punkte4], True, (0, 0, 255))
                p4_rec = math.sqrt(((p_m_1_x - p_m_2_x) ** 2) + ((p_m_1_y - p_m_2_y) ** 2))
                da = dist.euclidean((ploy_punkte4[0][0],ploy_punkte4[0][1]),(ploy_punkte4[1][0],ploy_punkte4[1][1]))
                try:

                   ""#print(p4_rec/ref)""

                except:pass

                #print(min_r)
                #print(min_y_1)
                min = cv.minAreaRect(con)
                points = cv.boxPoints(min)
                points = np.int0(points)
                cv.drawContours(frame,[points],0,(0,0,255),2)
                #print(points)
                #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text_punkt_min_con = points[1]
                p1_x = x + w / 2
                p1_y = y
                p2_x = (x+w/2)
                p2_y = y+h
                ploy_punkte3[0][0] = p1_x
                ploy_punkte3[0][1] = p1_y
                ploy_punkte3[1][0] = p2_x
                ploy_punkte3[1][1]= p2_y
                cv.polylines(frame,[ploy_punkte3],True,(0,0,255))
                p3_rec = math.sqrt(((p1_x - p2_x) ** 2) + ((p1_y - p2_y) ** 2))

                try:

                    p_dif_rec = p3_rec / min_punte_distanz_cm
                    text2 = cv.putText(frame, f"{p_dif_rec}", text_punkt_min_con, cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    #p_dif_rec2 = p4_rec / ref
                    print(p_dif_rec)
                    #print(p_dif_rec2 +2)

                except:
                    print("Error in min_punkte dis")


            else:
                pass
        else:
            pass
    else:
        pass

    cv.imshow("win2", th_img)
    cv.imshow("win",fr_gr)
    cv.imshow("bild1", frame)


    if cv.waitKey(1)== ord("q"):
        break
cap.release()
cv.destroyAllWindows()

