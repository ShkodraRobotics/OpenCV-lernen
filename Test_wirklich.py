import cv2 as cv
import numpy as np


cap = cv.VideoCapture(2)

circ = np.zeros((4,2),int)
count = 0
var = np.zeros((2,2),int)
var2 = np.full((4,2),-1,dtype=int)
print(var2)