import os

# comment out below line to enable tensorflow outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string("framework", "tf", "(tf, tflite, trt")
flags.DEFINE_string("weights", "./checkpoints/yolov4-416", "path to weights file")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny")
flags.DEFINE_string("model", "yolov4", "yolov3 or yolov4")
flags.DEFINE_list("images", "./data/images/kite.jpg", "path to input image")
flags.DEFINE_string("output", "./detections/", "path to output folder")
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.50, "score threshold")
flags.DEFINE_boolean("count", False, "count objects within images")
flags.DEFINE_boolean("dont_show", False, "dont show image output")
flags.DEFINE_boolean("info", False, "print info on detections")
flags.DEFINE_boolean("crop", False, "crop detections from images")
flags.DEFINE_boolean("ocr", False, "perform generic OCR on detection regions")
flags.DEFINE_boolean("plate", False, "perform license plate recognition")

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

weights = "./checkpoints/carno-416"
input_size = 416
model = "yolov4"
iou = 0.45
score = 0.50
input_ocr = True
crop = False
plate = False
info = False
input_count = False
dont_show = True
input = "./temp/uploads/"
output = "./temp/detections/"

# load model
saved_model_loaded = tf.saved_model.load(
    weights, tags=[tag_constants.SERVING]
)

def run(image_path):
    global weights
    global input_size
    global model
    global input_ocr
    global iou
    global score
    global crop
    global plate
    global info
    global input_count
    global dont_show
    global output

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.0

    # get image name by using split method
    image_name = image_path.split("/")[-1]
    image_name = image_name.split(".")[0]

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures["serving_default"]
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    (
        boxes,
        scores,
        classes,
        valid_detections,
    ) = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
        ),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score,
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    # hold all detection data in one variable
    pred_bbox = [
        bboxes,
        scores.numpy()[0],
        classes.numpy()[0],
        valid_detections.numpy()[0],
    ]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())

    # custom allowed classes (uncomment line below to allow detections for only people)
    # allowed_classes = ['person']

    # if crop flag is enabled, crop each detection and save it as new image
    if crop:
        crop_path = os.path.join(os.getcwd(), "detections", "crop", image_name)
        try:
            os.mkdir(crop_path)
        except FileExistsError:
            pass
        crop_objects(
            cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            pred_bbox,
            crop_path,
            allowed_classes,
        )

    # if ocr flag is enabled, perform general text extraction using Tesseract OCR on object detection bounding box
    licensePlate = ""
    if input_ocr:
        licensePlate = ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, image_name)

    # if count flag is enabled, perform counting of objects
    if input_count:
        # count objects found
        counted_classes = count_objects(
            pred_bbox, by_class=False, allowed_classes=allowed_classes
        )
        # loop through dict and print
        for key, value in counted_classes.items():
            print("Number of {}s: {}".format(key, value))
        image = utils.draw_bbox(
            original_image,
            pred_bbox,
            info,
            counted_classes,
            allowed_classes=allowed_classes,
            read_plate=plate,
        )
    else:
        image = utils.draw_bbox(
            original_image,
            pred_bbox,
            info,
            allowed_classes=allowed_classes,
            read_plate=plate,
        )

    image = Image.fromarray(image.astype(np.uint8))
    if not dont_show:
        image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(output + image_name + ".png", image)

    result = {
        "licensePlate": licensePlate,
        "fileName": image_name + ".png"
    }

    return result
