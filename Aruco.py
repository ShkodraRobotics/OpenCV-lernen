import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

aruco_dic = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

aruco_mark = cv.aruco.drawMarker(dictionary=aruco_dic, id=2,sidePixels=38)

cv.imwrite("aruco_4x4_3.jpg",aruco_mark)

cv.waitKey(0)

