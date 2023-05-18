import cv2 as cv
gt_file_path = "Original GTSRB Dataset\TrainIJCNN2013\\train_data\gt.txt"
save_path = "Edited GTSRB Dataset\\train_img\labels\\"
img_file_path = "Edited GTSRB Dataset\\train_img\images\\"
with open(gt_file_path) as f:
    gt_lines = f.readlines()

f.close()
for line in gt_lines :
    features = line.split(";")
    name = features[0][:-4]
    x1 = int(features[1])
    y1 = int(features[2])
    x2 = int(features[3])
    y2 = int(features[4])
    class_id = features[5].replace("\n", "")

    img = cv.imread(img_file_path+name+".jpg")

    shape_y,shape_x, =img.shape[:2]
    """print(shape_x,shape_y)
    image = cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
    cv.imshow(",img",img)
    cv.waitKey(0)"""




    f = open(save_path+name+".txt", 'a')
    f.write(class_id+" "+str(x1 / shape_x) +" "+str(y1 / shape_y)+" "+str(x2 / shape_x)+" "+str(y2 / shape_y)+"\n")
