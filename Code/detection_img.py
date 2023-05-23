from ultralytics import YOLO
import os
import cv2 as cv
model = YOLO("models\\baseline_model.pt")
file_path = "Edited GTSRB Dataset\\test_img"
result_path = "Results\\baseline_model"

for i in os.listdir(file_path):

    model.predict(source=file_path+"\\"+i, show=False, save=True)

