"""
Due to the limited number of images in the GTSDB dataset, I developed a Python script specifically for data augmentation.
This script allows for the creation of additional data by applying various augmentation techniques to the existing images in the
train and validation sets. Specifically, it generates five different augmented versions of each image, randomly assigning four of
them to the train folder and one to the validation folder. By augmenting the dataset in this manner, it helps to increase the diversity
 and quantity of available training data for improved model performance.

"""


import cv2 as cv
from Code.data_aug import *
from Code.bbox_util import *
import numpy as np
import os
import random
from PIL import Image

def change_format(list,img_shape):
    y_size,x_size = img_shape
    result = []
    for i in list :
        x1,y1,x2,y2,c = i[0]/x_size,i[1]/y_size,i[2]/x_size,i[3]/y_size,int(i[4])
        result.append(f"{c} {x1} {y1} {x2} {y2}")
    return result


img_path ="Edited GTSRB Dataset\\train_img\images"
box_path ="Edited GTSRB Dataset\\train_img\labels"
target_path = "Augmented_GTSRB_data"


for file in os.listdir(img_path):
    filename = file[:-4]
    img = cv.imread(img_path+"\\"+file)[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
    with open(box_path+"\\"+filename+".txt") as f:
        lines = f.readlines()
    f.close()
    bboxes = list()
    y,x = img.shape[:2]
    for i in lines:
        x1,y1,x2,y2,c = (float(i.split()[1])*x,float(i.split()[2])*y,float(i.split()[3])*x,float(i.split()[4])*y,int(i.split()[0]))
        bboxes.append([x1,y1,x2,y2,c])
    bboxes = np.array(bboxes)

    #Random Scale Part
    img_scale, bboxes_scale = RandomScale(0.3, diff = True)(img.copy(), bboxes.copy())

    #Random Translate Part
    img_translate, bboxes_translate = (None,None)
    while True:
        img_translate, bboxes_translate = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
        if len(bboxes_translate) != 0:
            break
        else: continue

    #Random Shear Part
    img_shear, bboxes_shear = RandomShear(0.2)(img.copy(), bboxes.copy())

    # Random Resize Part
    img_resize, bboxes_resize = Resize(608)(img.copy(), bboxes.copy())
    all_data = {"original":(img,bboxes),"scale":(img_scale,bboxes_scale),"shear":(img_shear,bboxes_shear),"translate" : (img_translate,bboxes_translate),"resize" : (img_resize,bboxes_resize)}
    val_key,val_data = random.choice(list(all_data.items()))
    train_data = {k: all_data[k] for k in set(list(all_data.keys())) - set([val_key])}

    #save augmented data
    val_path = "Augmented_GTSRB_data\Validation"
    train_path = "Augmented_GTSRB_data\Train"
    for key in train_data.keys():
        image_path = train_path+"\\images\\"+filename+"_"+key+".jpg"
        img_= train_data[key][0]
        im = Image.fromarray(img_)
        im.save(image_path)
        with open(train_path + "\\labels\\" + filename+"_"+key+".txt", "w") as f:
            #print(change_format(train_data[key][1], train_data[key][0].shape[:2]))
            f.writelines(change_format(train_data[key][1], train_data[key][0].shape[:2]))
        f.close()

    image_path = val_path+"\\images\\"+filename+"_"+val_key+".jpg"
    img_= val_data[0]
    im = Image.fromarray(img_)
    im.save(image_path)
    with open(val_path + "\\labels\\" + filename+"_"+val_key+".txt", "w") as f:
        #print(change_format(train_data[key][1], train_data[key][0].shape[:2]))
        f.writelines(change_format(val_data[1], val_data[0].shape[:2]))
    f.close()
