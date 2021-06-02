import cv2
import numpy as np


def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)

        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver


def rectContour(contour):
    rectCont = []
    for i in contour:
        area = cv2.contourArea(i)
        # print("Area:", area)

        if area > 50:
            per = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.01*per, True)
            # print("CP:", len(approx))
            if len(approx)==4:
                rectCont.append(i)
    
    rectCont = sorted(rectCont, key = cv2.contourArea , reverse = True)

    return rectCont


def getCornerPoint(cont):
    per = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.01*per, True)

    return approx

def reorder(points):
    points = points.reshape((4,2))
    add = points.sum(1)

    newPoints = np.zeros((4,1,2), np.int32)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints


def sliptBoxes(img):
    rows = np.vsplit(img, 5)
    boxes = []
    for row in rows:
        cols = np.hsplit(row,5)
        for box in cols:
            boxes.append(box)

    return boxes


def showAnswers(img, idx,grading,ans, questions, choices):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)

    for i in range(0, questions):
        myAns = idx[i]
        cX = (myAns*secW)+secW//2
        cY = (i*secH)+secH//2

        if grading[i] ==1:
            color = (0,255,0)
        else:
            color = (0,0,255)
            correctAns = ans[i]
            center = ((correctAns*secW)+secW//2, (i*secH)+secH//2)
            cv2.circle(img,center,30,(0,255,0), cv2.FILLED)


        cv2.circle(img,(cX, cY),50,color, cv2.FILLED)

    return img