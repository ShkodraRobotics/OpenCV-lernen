import cv2 as cv
import os
import keyboard

"""pfad = "D:\LL\Pictures\PYM\data"

try:
    os.mkdir(pfad)

except OSError:
    print(OSError)"""

cap = cv.VideoCapture(2)


frame_num = 0

"""def save(frame_num):
    
    name = "D:\LL\Pictures\PYM\data\\bild" + str(frame_num) + ".jpg"

    cv.imwrite(name, fram)

    frame_num += 1
    return save(frame_num)"""
ret = cap.set(3, 1080)
ret = cap.set(4, 1920)

while True:
   ret, fram = cap.read()



   cv.imshow("Fram", fram)

   k = cv.waitKey(1)

   if k== ord("w"):
       name = "D:\LL\Pictures\PYM\data\\bild" + str(frame_num) + ".jpg"

       cv.imwrite(name, fram)

       frame_num += 1


   elif k == ord("q"):
       break





cap.release()
cv.destroyAllWindows()
