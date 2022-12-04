import os
import re
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg
import imutils
from PIL import Image
from datetime import datetime
from itertools import combinations
import math

gray_pre_output = "./temp/graypre/"
edged_pre_output = "./temp/edgedpre/"
warp_output = "./temp/warp/"
smoothened_output = "./temp/smoothened/"
thresh_output = "./temp/thresh/"
erosion_output = "./temp/erosion/"
blur_output = "./temp/blur/"

def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    print(rect)
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    print(width_a)
    print(width_b)

    print(max_width)
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    print(height_a)
    print(height_b)

    max_height = max(int(height_a), int(height_b))
    print(max_height)

    dist = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    print(dist)
    perspect = cv2.getPerspectiveTransform(rect, dist)
    warped = cv2.warpPerspective(image, perspect, (max_width, max_height))
    return warped


# function to count objects, can return total classes or count per class
def count_objects(
    data,
    by_class=False,
    allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values()),
):
    boxes, scores, classes, num_objects = data

    # create dictionary to hold count of objects
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
        counts["total object"] = num_objects

    return counts


# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    # create dictionary to hold count of objects for image name
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
            cropped_img = img[
                int(ymin) - 5 : int(ymax) + 5, int(xmin) - 5 : int(xmax) + 5
            ]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + "_" + str(counts[class_name]) + ".png"
            img_path = os.path.join(path, img_name)
            # save image
            cv2.imwrite(img_path, cropped_img)
        else:
            continue


# function to run general Tesseract OCR on any detections
def ocr(img, data, image_name):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection

        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin) - 30 : int(ymax) + 30, int(xmin) - 30 : int(xmax) + 30]
        orgi = box.copy()
        # ratio = box.shape[0] / (300*0.975)
        # img1 = imutils.resize(box, width=300)
        # alpha = 1 # Contrast control (1.0-3.0)
        # beta = 30 # Brightness control (0-100)

        # img1 = cv2.convertScaleAbs(img1, alpha=alpha, beta=beta)

        # cv2.imshow('manual_result', img1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        # grayscale region within bounding box
        gray_pre = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        gray_pre = cv2.GaussianBlur(gray_pre, (5, 5), 0)
        cv2.imwrite(gray_pre_output + image_name + ".png", gray_pre)
        # cv2.imshow("greyed image", gray_pre)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        edged_pre = cv2.Canny(gray_pre, 30, 200)
        # thresh_pre= cv2.adaptiveThreshold(edged_pre,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        # thresh = cv2.threshold(filter, 122, 255, cv2.THRESH_BINARY_INV)[1]

        cv2.imwrite(edged_pre_output + image_name + ".png", edged_pre)
        # cv2.imshow("grey image", edged_pre)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)

        contours = cv2.findContours(
            edged_pre.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # cv2.imshow("grey image", contours)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)

        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            print(len(approx))

            if len(approx) == 4:
                cnt = approx
                break

        # print("STEP 2: Find contours of paper")
        # cv2.drawContours(img1, [cnt], -1, (0, 255, 0), 2)
        # cv2.imshow("Outline", img1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cnt = cnt.reshape(4, 2)

        warp = point_transform(orgi, cnt)

        cv2.imwrite(warp_output + image_name + ".png", warp)
        # cv2.imshow("warp image", warp)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
       
        # 柔和
        new_warp = imutils.resize(warp, width=300)
        gray = cv2.cvtColor(new_warp, cv2.COLOR_BGR2GRAY)

        filter = cv2.bilateralFilter(gray, 5, 50, 50)

        

        cv2.imwrite(smoothened_output + image_name + ".png", filter)
        # cv2.imshow("smoothened image", filter)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        # 二直化
        thresh = cv2.threshold(filter, 122, 255, cv2.THRESH_BINARY_INV)[1]

        cv2.imwrite(thresh_output + image_name + ".png", thresh)
        # cv2.imshow("smoothened image", thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        # select the text
        print(thresh)
        (rows, cols) = thresh.shape
        h_projection = np.array([x / rows for x in thresh.sum(axis=0)])
        threshold = (np.max(h_projection) - np.min(h_projection)) / 10
        print("we will use threshold {} for horizontal".format(threshold))

        # select the black areas 水平
        black_areas = np.where(h_projection < threshold)
        for j in black_areas:
            thresh[:, j] = 0

        v_projection = np.array([x / cols for x in thresh.sum(axis=1)])
        threshold = (np.max(v_projection) - np.min(v_projection)) / 3.8
        print("we will use threshold {} for vertical".format(threshold))

        black_areas = np.where(v_projection < threshold)
        # select the black areas 垂直
        for j in black_areas:
            thresh[j, :] = 0


        kernel = np.ones((2,3),np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations=1)
        cv2.imwrite(erosion_output + image_name + ".png", erosion)
        # cv2.imshow("Top 30 contours", erosion)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)

        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(erosion, 3)
        # cv2.imshow("Top 30 contours", blur)

        # RGBimage = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        # PILimage = Image.fromarray(RGBimage)
        # PILimage.save(f"./pre_img/{str(datetime.now())}.png", dpi=(300, 300))
        
        cv2.imwrite(blur_output + image_name + ".png", blur)
        # cv2.imshow("Top 30 blur",blur )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        # resize image to double the original size as tesseract does better with certain text size
        # blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(
                blur,
                lang="eng",
                config="-c tessedit_char_whitelist=-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 11 --oem 3",
            )
            print("Class: {}, Text Extracted: {}".format(class_name, text))
            return text
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_lineno)
            return ""
