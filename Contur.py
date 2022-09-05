import numpy as np
import cv2 as cv

cap = cv.VideoCapture(2)


"""def contur(kon):
    größte = np.array([])
    max_fla = 0
    for i in kon:
        fla = cv.contourArea(i)
        if fla > 5000:
            peri = cv.arcLength(i,True)
            aprox = cv.approxPolyDP(i,0.02 + peri, True)
            if fla > max_fla :
                größte = aprox
                max_fla = fla
    return größte , max_fla"""







while True:
    ret ,frame = cap.read()

    frame_gray = cv.cvtColor(frame,cv.COLOR_BGRA2GRAY)
    rett,thresh = cv.threshold(frame_gray,150,255,cv.THRESH_BINARY)
    thrersh_ar = np.array(thresh)

    con, hih = cv.findContours(thrersh_ar,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    #größte , max_fla = contur(con)

    #print(max_fla)

    for idx in con:
        kon = cv.contourArea(idx)
        if kon > 5000:
            x, y, w, h = cv.boundingRect(idx)
            x_c = x+(w/2)
            y_c= y+(h/2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.circle(frame,(int(x_c),int(y_c)),5,(0,0,255),-1)
            print(x,y)

    #cv.drawContours(frame,größte,-1,(0,0,255),3)

    cv.imshow("Thr",thresh)
    cv.imshow("Frame",frame)

    if cv.waitKey(1) == ord("q"):
        break