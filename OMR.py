# bibliotecas necessarias
import cv2 
import numpy as np
from numpy.lib import utils
import utils

###############################
path = 'Imgs/1.jpg'

# tamanho da imagem
wImg,hImg = 700, 700
#numero de questoes
#numeros de escolhas possiveis
questions, choices = 5, 5
# respostas corretas
ans = [1,2,0,1,4]

dim = (wImg, hImg)
###############################

#lendo a imagem
img = cv2.imread(path)

## IMAGE PREPROCESSING
img = cv2.resize(img, dim)
imgCont = img.copy() 
imgFinal = img.copy() 
imgBigCont = img.copy() # copping the img for do the contour
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

## FINDING THE CONTOURS
cont,hier = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgCont, cont, -1, (255,0,0),10)


## FINDING RECTANGLES
rectCont = utils.rectContour(cont)
bigSqr = utils.getCornerPoint(rectCont[0]) # defining the biggest contour 
scdSqr = utils.getCornerPoint(rectCont[1]) # defining the second biggest contour 

if bigSqr.size != 0 and scdSqr.size != 0:
    cv2.drawContours(imgBigCont, bigSqr, -1, (0,255,0),15)
    cv2.drawContours(imgBigCont, scdSqr, -1, (0,0,255),15)

    bigSqr = utils.reorder(bigSqr)
    scdSqr = utils.reorder(scdSqr)

    pt1 = np.float32(bigSqr)
    pt2 = np.float32([[0,0],[wImg, 0],[0,hImg], [wImg, hImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)

    pt3 = np.float32(scdSqr)
    pt4 = np.float32([[0,0],[325, 0],[0,150], [325, 150]])
    matrixScd = cv2.getPerspectiveTransform(pt3, pt4)

    imgWarpPersp = cv2.warpPerspective(img, matrix, (wImg, hImg))
    imgWarpPerspGrade = cv2.warpPerspective(img, matrixScd, (325, 150))

    # APPLY THRESHOLH
    imgWarpGray = cv2.cvtColor(imgWarpPersp, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

    boxes = utils.sliptBoxes(imgThresh)

    # GETTING THE NON ZERO PIXELS VALUES
    pixelsValues = np.zeros((questions,choices))
    countC, countR = 0,0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        pixelsValues[countR][countC] = totalPixels
        countC += 1

        if (countC == choices): countR += 1; countC=0 

    # FINDING IDX VALUES
    idx = []
    for x in range(0,questions):
        arr = pixelsValues[x]
        idxVal = np.where(arr==np.amax(arr))
        
        idx.append(idxVal[0][0])

    # GRADING
    grading = []
    for i in range(0,questions):
        if ans[i] == idx[i]:
            grading.append(1)
        else:
            grading.append(0)

    # FINAL GRADE
    score = (sum(grading)/questions) * 100

    # DISPLAYING ANSWERS
    imgResult = imgWarpPersp.copy()
    imgResult =  utils.showAnswers(imgResult, idx, grading, ans, questions, choices)
    imgRawDrawing = np.zeros_like(imgWarpPersp)
    imgRawDrawing =  utils.showAnswers(imgRawDrawing, idx, grading, ans, questions, choices)

    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (wImg, hImg))


    imgRawGrade = np.zeros_like(imgWarpPerspGrade)
    cv2.putText(imgRawGrade, str( int(score)), (100, 100), cv2.FONT_HERSHEY_TRIPLEX, 3 , (255,255,255),5)
    invMatrixScd = cv2.getPerspectiveTransform(pt4, pt3)
    imgInvGrade = cv2.warpPerspective(imgRawGrade, invMatrixScd, (wImg, hImg))

    
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 2 ,0)
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGrade, -1 ,0)

# Para ver as transformacoes feitas na imagem para chegar ao resultado final
# basta descomentar esta parte do codigo
# imgArray = ([img,imgGray,imgBlur,imgCanny],
#            [imgCont, imgBigCont, imgWarpPersp, imgThresh],
#            [imgResult, imgRawDrawing, imgInvWarp, imgFinal])

# labels = [['ORIGINAL', 'GRAY', "BLUR", "CANNY"],
#            ['CONTOURS', 'BIGGEST CONTOURS', 'WARP', "THRESHOLD"],
#            ['RESULT', 'RAW DRAWING', 'INVERSE WARP', 'FINAL' ] ]
# imgStacked = utils.stackImages(imgArray, 0.4,  labels)
#cv2.imshow("Stacked Images", imgStacked)

cv2.imshow('Final', imgFinal)
cv2.waitKey(0)
