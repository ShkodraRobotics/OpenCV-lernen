import cv2 as cv
import numpy as np
import tkinter as tk
import math
from PIL import Image, ImageTk
import pyfirmata


root = tk.Tk()
root.title("Titel")

img = tk.Frame(root,width=600,height=600)
img.grid(row=0,column=0,padx=10,pady=10)


vmain = tk.Label(img)
vmain.grid(row=0, column=0, padx=10, pady=2)

cap = cv.VideoCapture(1)
aruco_dic = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
aruco_dic2 = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)
aruco_dic3 = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_50)

ploy_punkte = np.zeros((3,2),int)
ploy_punkte2 = np.zeros((2,2),int)
tex_ar = np.zeros((1,2),int)
cont = 0
global pin9

try:

    board=pyfirmata.Arduino('COM4')

    iter8 = pyfirmata.util.Iterator(board)
    iter8.start()

    pin9 = board.get_pin('d:5:s')
except:

    print("Keine Verbindung mit Arduino")



def move_servo(angle):
    pin9.write(angle)




def win():
    ret, frame = cap.read()

    #frame = cv.resize(frame, (1020, 720))

    fr_gr = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)

    corner, ids, rej_cor = cv.aruco.detectMarkers(fr_gr, aruco_dic)
    corner2, ids2, rej_cor2 = cv.aruco.detectMarkers(fr_gr, aruco_dic2)
    corner3, ids2, rej_cor2 = cv.aruco.detectMarkers(fr_gr, aruco_dic3)

    # print(corner,ids)
    # aruco_frame = cv.aruco.drawDetectedMarkers(image=frame,corners=corner,ids=ids,borderColor=(0,0,255))
    h, w = 0, 0

    for idx in corner:
        x, y, h, w = cv.boundingRect(idx)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[0] = x + (w / 2), y + (h / 2)
    for idx2 in corner2:
        x, y, h, w = cv.boundingRect(idx2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[1] = x + (w / 2), y + (h / 2)
    for idx3 in corner3:
        x, y, h, w = cv.boundingRect(idx3)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ploy_punkte[2] = x + (w / 2), y + (h / 2)
        tex_ar = x,y

    ploy_punkte2[0] = ploy_punkte[0]
    ploy_punkte2[1] = ploy_punkte[2]
    # print(w)
    p1x = ploy_punkte2[0][0]
    p1y = ploy_punkte2[0][1]
    p2x = ploy_punkte2[1][0]
    p2y = ploy_punkte2[1][1]
    p3 = math.sqrt(((p1x - p2x) ** 2) + ((p1y - p2y) ** 2))
    pol = cv.polylines(frame, [ploy_punkte], False, (0, 0, 255), 5)
    pol2 = cv.polylines(frame, [ploy_punkte2], False, (0, 255, 0))
    try:

        p_dif = p3 / w
        p_dif = p_dif - 2
        print(round(p_dif))

    except:
        pass
    try:
        text = cv.putText(frame, f"{round(p_dif,2)} cm", (tex_ar), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 2)
    except:
        pass


    rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGBA)
    imgar = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image = imgar)
    vmain.imgtk = imgtk
    vmain.configure(image=imgtk)
    vmain.after(10,win)

scale = tk.Scale(root, command = move_servo,from_=20, to = 100,
                orient = tk.HORIZONTAL, length = 400, label = 'Angle')
scale.grid(row=1,column=0)

win()
root.mainloop()