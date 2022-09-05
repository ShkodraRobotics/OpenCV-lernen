
import cv2 as cv
import numpy as np

image = cv.imread("road.jpg",1)


h,w = image.shape[:2]


print(f"{h}"f"{w}")

rese = cv.resize(image,(0,0),fy=0.33,fx=0.33)

rec = cv.rectangle(rese,(600,300),(700,500),(255,0,0),2)
tex = cv.putText(rec,"TEXTZ",(600,280),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

centre = (w/2,h/2)
matrix = cv.getRotationMatrix2D(centre,-45,1.0)
rotator = cv.warpAffine(image,matrix,(w,h))

"""ratio = 800/w

dim = (800,int(h*ratio))
rese = cv.resize(image,dim)"""

cv.imshow("img",tex)
cv.waitKey(0)







