from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

model = YOLO("models\\baseline_model.pt")

#cap = cv.VideoCapture("..\Videos\ppe-2.mp4")

cap = cv.VideoCapture("Video_Data\\video-1.mp4")
cap.set(3,640)
cap.set(4,640)

"""classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]"""

"""classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
"""
classNames =  ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70', 'NO-speed limit 80', 'restriction ends 80', 'speed limit 100', 'speed limit 120',
'no overtaking', 'no overtaking (trucks)', 'priority at next intersection', 'priority road', 'give way', 'stop', 'no traffic both ways', 'no trucks', 'no entry', 'danger',
'bend left', 'bend right', 'bend', 'uneven road', 'slippery road', 'road narrows','construction', 'traffic signal', 'pedestrian crossing', 'school crossing', 'cycles crossing',
'snow', 'animals', 'restriction ends', 'go right', 'go left', 'go straight', 'go right or straight', 'go left or straight', 'keep right', 'keep left', 'roundabout',
'restriction ends (overtaking)', 'restriction ends (overtaking (trucks))']
while True:
    success, img = cap.read()
    result = model(img,stream=True)
    for r in result:
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            """x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cv.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3,)"""

            x,y,w,h = x1,y1,x2-x1,y2-y1

            bbox= (int(x),int(y),int(w),int(h))

            cvzone.cornerRect(img,bbox)

            conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]
            cvzone.putTextRect(img, f'{conf} {classNames[int(cls)]}', pos = (int(max(0, x1)), int(max(35, y1))), scale=1, thickness=1)


    cv.imshow("img",img)
    cv.waitKey(1)

