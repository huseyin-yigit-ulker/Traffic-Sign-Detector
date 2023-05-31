"""
   Performs object detection on a video using the YOLO model and displays the results.

   Args:
       model (YOLO): The YOLO object detection model.
       video_path (str): The path to the input video.
"""

from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

# Set the path to the YOLO model and input video
model_path = "../Results/model_4/best.pt"
video_path = "../Video_Data/test1.mp4"

model = YOLO(model_path)

#cap = cv.VideoCapture("..\Videos\ppe-2.mp4")

cap = cv.VideoCapture(video_path)
cap.set(3,640)
cap.set(4,640)

"""classNames =  ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70', 'NO-speed limit 80', 'restriction ends 80', 'speed limit 100', 'speed limit 120',
'no overtaking', 'no overtaking (trucks)', 'priority at next intersection', 'priority road', 'give way', 'stop', 'no traffic both ways', 'no trucks', 'no entry', 'danger',
'bend left', 'bend right', 'bend', 'uneven road', 'slippery road', 'road narrows','construction', 'traffic signal', 'pedestrian crossing', 'school crossing', 'cycles crossing',
'snow', 'animals', 'restriction ends', 'go right', 'go left', 'go straight', 'go right or straight', 'go left or straight', 'keep right', 'keep left', 'roundabout',
'restriction ends (overtaking)', 'restriction ends (overtaking (trucks))']"""

classNames =['ANIMALS', 'CONSTRUCTION', 'CYCLES CROSSING', 'DANGER', 'NO ENTRY', 'PEDESTRIAN CROSSING', 'SCHOOL CROSSING', 'SNOW', 'STOP', 'bend left', 'bend right', 'bend', 'give way', 'go left or straight', 'go left', 'go right or straight', 'go right', 'go straight', 'keep left', 'keep right', 'no overtaking (trucks)', 'no overtaking -trucks-', 'no overtaking', 'no traffic both ways', 'no trucks', 'priority at next intersection', 'priority road', 'restriction ends (overtaking (trucks))', 'restriction ends (overtaking)', 'restriction ends -overtaking -trucks--', 'restriction ends -overtaking-', 'restriction ends 80', 'restriction ends', 'road narrows', 'roundabout', 'slippery road', 'speed limit 100', 'speed limit 120', 'speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70', 'speed limit 80', 'traffic signal', 'uneven road']

while True:
    success, img = cap.read()
    result = model(img,stream=True)
    for r in result:
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x,y,w,h = x1,y1,x2-x1,y2-y1
            bbox= (int(x),int(y),int(w),int(h))

            conf = math.ceil((box.conf[0]*100))/100
            cls = box.cls[0]
            if conf > 0.5 :
                cvzone.cornerRect(img, bbox)
                cvzone.putTextRect(img, f'{conf} {classNames[int(cls)]}', pos = (int(max(0, x1)), int(max(35, y1))), scale=1, thickness=1)


    cv.imshow("img",img)
    cv.waitKey(1)

