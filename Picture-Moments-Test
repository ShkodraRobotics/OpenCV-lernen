import cv2 as cv
import numpy as np

cap = cv.VideoCapture(2)
img1 = cv.imread("TT.jpg")
#cv.resize(cap)

img = cv.resize(img1,(0,0),fx=0.3,fy=0.3)
def no(x):
    pass

cv.namedWindow("bild")
cv.createTrackbar("thresh","bild",0,255,no)



count = 0
ret = cap.set(3, 1080)
ret = cap.set(4, 1920)
while True:
    ret , frame = cap.read()

    #frame = cv.resize(frame,(0,0),fx=2,fy=2)

    im_gray = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGRA2GRAY)

    t = cv.getTrackbarPos("thresh","bild")

    rett,thresh_frame = cv.threshold(gray_frame,t,255,cv.ADAPTIVE_THRESH_MEAN_C)
    thresh_frame =np.array(thresh_frame)

    rot ,im_thre = cv.threshold(im_gray,t,255,cv.ADAPTIVE_THRESH_MEAN_C)
    # = cv.selectROI()

    con , hir = cv.findContours(thresh_frame,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)


    for idx in con:

        Fl = cv.contourArea(idx)
        if Fl > 10000:
            con_dr = cv.drawContours(frame,idx,-1,(0,0,255),3)
            M = cv.moments(idx)
            hu = cv.HuMoments(M)
            count += 1
            for i in range(0, 7):
                hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(abs(hu[i]))

        else:
            pass





    cv.imshow("bild",frame)
    cv.imshow("bild2",im_thre)

    k = cv.waitKey(1)
    if k == ord("w"):



        print(M)
        print(hu)
    elif k == ord("q"):
        break


cap.release()
cv.destroyAllWindows()
