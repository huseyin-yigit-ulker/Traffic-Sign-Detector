"""
  Preprocesses the data by reading images, filtering annotations, and saving the images in a new directory.
  The provided Python script processes files in the .ppm format and converts them to the .jpg format.

  Args:
      path_dir (str): The path to the directory containing the images to be processed.
      save_path (str): The path to the directory where the preprocessed images will be saved.
      ground_truth_path (str): The path to the ground truth file containing the annotations.

  Returns:
      train_data_df (pd.DataFrame): The preprocessed data containing the filtered annotations in a DataFrame.
"""

import pandas as pd
import cv2 as cv
import os
path_dir = "TestIJCNN2013Download\\"
save_path = "test_img"
#reading data guides
ground_truth_path =  "TrainIJCNN2013\\train_data\\gt.txt"
data = pd.read_csv(ground_truth_path,sep=';',names=['path','left','top','right','bottom','id'])
train_data_df = pd.DataFrame(columns=data.columns)
for current_dir, dirs, files in os.walk(path_dir):
    for f in sorted(files):
        image_name = f[:-4]
        img = cv.imread(path_dir + f)
        single_yolo_dat = data.loc[data['path'] == f].copy()
        # and, in this way, initial dataFrame will not be changed
        # Checking if there is no any annotations for current image
        if single_yolo_dat.isnull().values.all():
            # Removing this image from train data
            # print(f)
            os.remove(path_dir + '/' + f)

        # Now save the resulted_frame to a folder inside path_dir
        else:
            train_data_df = train_data_df.append(single_yolo_dat)
            # Now writng and saving the image from ppm format to jpg format using OpenCV
            cv.imwrite(save_path + '/' + image_name + '.jpg', img)
train_data_df = train_data_df[~train_data_df.index.duplicated(keep='first')];
train_data_df.sort_index(inplace=True);