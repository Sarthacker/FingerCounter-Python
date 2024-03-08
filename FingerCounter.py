import cv2 as cv
import time
import mediapipe as mp
import HandTrackingModule as htm
import os

wcam, hcam = 640, 480
ptime=0

cap = cv.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

folder = "Fingers"
myList = os.listdir(folder)
imgList = []
for imgs in myList:
    image = cv.imread(f'{folder}/{imgs}')
    if image is not None:  # Check if the image is valid
        image = cv.resize(image, (200,200))  # Resize the image
        imgList.append(image)
    else:
        print(f"Unable to read image: {imgs}")

detector=htm.handDetector(detectionCon=0.75)

tipIds=[4,8,12,16,20] # the value of thqe tip of each of the finger is stored int his list except thumb which serves a special case

while True:
    success, img = cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False) # this is the list of the landmarks of the hands that appear on the camera eventually

    if lmList:
        fingers=[]

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers=fingers.count(1)
        # print(totalFingers)

        if imgList:  # Check if imgList is not empty
            img_height, img_width, c = imgList[totalFingers-1].shape
            if img_height <= 200 and img_width <= 200:
                img[0:img_height, 0:img_width] = imgList[totalFingers-1]
            else:
                print("Image from imgList is larger than the region of interest")

        cv.rectangle(img,(20,255),(170,425),(0,255,0), cv.FILLED)
        cv.putText(img,str(totalFingers),(45,375),cv.FONT_HERSHEY_PLAIN,10,(255,0,0),25)

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv.putText(img, f'FPS: {int(fps)}', (400, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv.imshow("Image", img)

    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
