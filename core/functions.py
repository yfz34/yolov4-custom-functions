import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg

# function to count objects, can return total classes or count per class
def count_objects(data, by_class = False, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts

# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name )
            # save image
            cv2.imwrite(img_path, cropped_img)
        else:
            continue
        
# function to run general Tesseract OCR on any detections 
def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection

        
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin)+30:int(ymax)-15, int(xmin)+30:int(xmax)-30]


        cv2.imshow("original image", box)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)

        cv2.imshow("greyed image", box)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # 柔和
        filter = cv2.bilateralFilter(gray, 10, 90, 90)
        cv2.imshow("smoothened image", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)



        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(filter, 122, 255, cv2.THRESH_BINARY_INV)[1]

       

        # select the text
        (rows, cols) = thresh.shape
        h_projection = np.array([x / rows for x in thresh.sum(axis=0)])
        threshold = (np.max(h_projection) - np.min(h_projection)) / 10
        print("we will use threshold {} for horizontal".format(threshold))

        # select the black areas 水平
        black_areas = np.where(h_projection < threshold)
        for j in black_areas:
            thresh[:, j] = 0


        v_projection = np.array([x / cols for x in thresh.sum(axis=1)])
        threshold = (np.max(v_projection) - np.min(v_projection)) / 4
        print("we will use threshold {} for vertical".format(threshold))

        black_areas = np.where(v_projection < threshold)
        # select the black areas 垂直
        for j in black_areas:
            thresh[j, :] = 0
        
        cv2.imshow("Top 30 contours", thresh)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)





        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)

        cv2.imshow("Top 30 contours", thresh)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # resize image to double the original size as tesseract does better with certain text size
        # blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except: 
            text = None