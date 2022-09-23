import cv2
import numpy as np
import time

def set_tbar(a):
    pass

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("TrackBars")
cv2.createTrackbar("Focus", "TrackBars", 0, 255, set_tbar)

myCanvas = np.zeros((1280, 720, 3), np.uint8)
myPoints = []

drawing = False
penDetected = False
xp, yp = 0, 0

def findPen(cap):
    cap = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
    lower = np.array([95,150,20])
    upper = np.array([115,255,255])
    mask = cv2.inRange(cap, lower, upper)
    x, y = findContours(mask)
    cv2.circle(img, (x,y), 5, (255,0,0), cv2.FILLED) #marker point position
    return mask, x, y

def findContours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            # cv2.drawContours(img, cnt, -1, (255,0,0), 3) #draw contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            penDetected = True
    return x, y+h//2

while True:
    focus = cv2.getTrackbarPos("Focus", "TrackBars")
    cap.set(28, focus)

    start = time.time()

    success, img = cap.read()
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    mask, x, y = findPen(img) #show mask with pen
    mask = cv2.medianBlur(mask, 5)

    if cv2.waitKey(32) & 0xFF == ord('z'):
        drawing = True
    else:
        drawing = False
        xp, yp = 0, 0

    if drawing is True:
        print("Drawing...")
        if xp == 0 and yp == 0:
            xp, yp = x, y
        if x == 0 and y == 0:
            pass
        else:
            cv2.line(myCanvas, (xp, yp), (x, y), (255, 0, 0), 3)
        xp, yp = x, y
    else:
        print("Not drawing")

    canvasGrey = cv2.cvtColor(myCanvas, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Demo canvas grey", canvasGrey)
    _, imgInv = cv2.threshold(canvasGrey, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Demo img inverted", imgInv)
    img = cv2.bitwise_and(img, imgInv)
    cv2.imshow("Demo img and inv", img)
    img = cv2.bitwise_or(img, myCanvas)

    # cv2.imshow("Camera", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Canvas", myCanvas)
    cv2.imshow("Final", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()