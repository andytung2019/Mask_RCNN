#encoding:utf-8
import time
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import csv


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE_SHAPES=np.array([[256, 256], [128, 128],  [ 64,  64],  [ 32,  32],  [ 16,  16]])
    BACKBONE_STRIDES=[4, 8, 16, 32, 64]
    BATCH_SIZE=1
    BBOX_STD_DEV=[ 0.1,  0.1,  0.2,  0.2]
    DETECTION_MAX_INSTANCES=100
    DETECTION_MIN_CONFIDENCE=0.6 #0.5
    DETECTION_NMS_THRESHOLD=0.3
    IMAGE_MAX_DIM=1024
    IMAGE_MIN_DIM=800
    IMAGE_PADDING=True
    IMAGE_SHAPE=np.array([1024, 1024,    3])
    LEARNING_MOMENTUM=0.9
    LEARNING_RATE =0.002
    MASK_POOL_SIZE=14
    MASK_SHAPE    =[28, 28]
    MAX_GT_INSTANCES=100
    MEAN_PIXEL      =[ 123.7,  116.8,  103.9]
    MINI_MASK_SHAPE =(56, 56)
    NAME            ="coco"
    NUM_CLASSES     =81
    POOL_SIZE       =7
    POST_NMS_ROIS_INFERENCE =1000
    POST_NMS_ROIS_TRAINING  =2000
    ROI_POSITIVE_RATIO=0.33
    RPN_ANCHOR_RATIOS =[0.5, 1, 2]
    RPN_ANCHOR_SCALES =(32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE =2
    RPN_BBOX_STD_DEV  =np.array([ 0.1,  0.1,  0.2 , 0.2])
    RPN_TRAIN_ANCHORS_PER_IMAGE=256
    STEPS_PER_EPOCH            =1000
    TRAIN_ROIS_PER_IMAGE       =128
    USE_MINI_MASK              =True
    USE_RPN_ROIS               =True
    VALIDATION_STPES           =50
    WEIGHT_DECAY               =0.0001



class MyCheck():

    def __init__(self):
        config = InferenceConfig()
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True, exclude =[
		"mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox","mrcnn_mask"
	])

        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle',
                       'bus', 'train', 'truck'
		      ]

        self.model = model
        self.config = config
        self.classes = class_names
        self.list_out = []
        self.list_pic = []

    #保存所有图片到列表
    def get_pic_list(self):
        pic_subfix = ['jpg', 'JPG', 'JPEG', 'jpeg', 'png']
        path = './images'
        path = os.path.expanduser(path)
        for (dirname, subdir, subfile) in os.walk(path):
            for f in subfile:
                sufix = os.path.splitext(f)[1][1:]
                if sufix in pic_subfix:
                    path = os.path.join(dirname, f)
                    dit= (f, path)
                    self.list_pic.append(dit)
        print('get pic list:%d',len(self.list_pic))


    #检测小批量图片：10
    def detect_pic_list(self, list_path):

        for name_path in list_path:
            pic_name = name_path[0]
            image = skimage.io.imread(name_path[1])
            results = self.model.detect([image])
            r = results[0]
            self.list_out.append({'image_name': pic_name, 'rois': r['rois'], 'class_ids': r['class_ids']})

    def write_to_csv(self):
        with open(r'out.csv', 'a') as out_csv:
            fields = ['image_name', 'rois', 'class_ids']
            for i in range(len(self.list_out)):
                r = self.list_out[i]
                writer = csv.DictWriter(out_csv, fieldnames=fields)
                writer.writerow({'image_name':r['image_name'], 'rois': r['rois'], 'class_ids': r['class_ids']})
  #检测所有图片
    def detect_all_pic(self):
        print("img list len", len(self.list_pic))

        start = 0
        end = 0
        while start < len(self.list_pic):
            #估算时间
            tm_start = time.clock()
            list_path = []

            start = end
            end = start + 10
            if end >= len(self.list_pic):
                end = len(self.list_pic)
            for idx in range(start, end):
                list_path.append(self.list_pic[idx])
            self.detect_pic_list(list_path,verbos=0)
            tm_end = time.clock()
            print("image from %d->%d time:%d", start, end, tm_end-tm_start)

mycheck = MyCheck()
mycheck.get_pic_list()
mycheck.detect_all_pic()
mycheck.write_to_csv()

