"""
    Performs object detection on images using the YOLO model and may saves the results.

    Args:
        model (YOLO): The YOLO object detection model.
        file_path (str): The path to the directory containing the input images.
        result_path (str): The path to the directory where the detection results will be saved.

"""


from ultralytics import YOLO
import os
import cv2 as cv

# Load YOLO model
model = YOLO("..\Models\\baseline_model.pt")

file_path = "..\Edited GTSRB Dataset\\test_img"

result_path = "Results\\baseline_model"

for i in os.listdir(file_path):
    model.predict(source=file_path+"\\"+i, show=True, save=True ,project = result_path)
    cv.waitKey(0)
    cv.destroyAllWindows()

