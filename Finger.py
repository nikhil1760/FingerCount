import cv2,time,os,HandTrackingModule as htm
cap=cv2.VideoCapture(0)
wCam,hCam=1000,3000
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0
folderPath="FingerImage"
myList=os.listdir(folderPath) # store the images in sorted order of images , files , folders
overlayList=[]
print(myList)
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}') # traces the path of images
    overlayList.append(image) # store the path in list

for i in overlayList:
    print(i.shape)
detector=htm.handDetector(detectionCon=0.75)
tipsIds=[4,8,12,16,20]
while True:
    fingers=[]
    success,img=cap.read()
    img=detector.findHands(img)
    lmlist=detector.findPosition(img,draw=False)

    if len(lmlist)!=0:
        if lmlist[tipsIds[0]][1] > lmlist[tipsIds[0] - 1][1]:  # refer to point figure right/left
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmlist[tipsIds[id]][2]<lmlist[tipsIds[id]-2][2]: # refer to point figure bottom/top
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        totalfinger=fingers.count(1)
        h,w,c=overlayList[totalfinger].shape
        img[0:h,0:w]=overlayList[totalfinger]
        cv2.rectangle(img, (20, 550), (170, 900), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalfinger), (45, 700), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("img", img)
    cv2.waitKey(1)
