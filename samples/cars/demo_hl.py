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
            results = self.model.detect([image], verbose=1)
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
            self.detect_pic_list(list_path)
            tm_end = time.clock()
            #print("image from %d->%d time:%d", start, end, tm_end-tm_start)

mycheck = MyCheck()
mycheck.get_pic_list()
mycheck.detect_all_pic()
mycheck.write_to_csv()


