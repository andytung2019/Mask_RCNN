import cv2 as cv
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime
import PIL.Image
import json


IMAGE_DIR = ('images\\')
INFO = {
    "description": "China Street Vehicle Dataset",
    "url": "https://github.com/andytung2019",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "andy tung",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 3,
        'name': 'car',
        'supercategory': 'vehicle',
    },
    {
        'id': 6,
        'name': 'bus',
        'supercategory': 'vehicle',
    },
    {
        'id': 8,
        'name': 'truck',
        'supercategory': 'vehicle',
    },
]


class Rect:
    idx = -1
    x = None
    y = None
    w = None
    h = None

    def str2float(self, s):
        def char2num(s):
            return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
            # 这事实上是一个字典

        index_point = s.find('.')
        if index_point == -1:
            daichu = 1
        else:
            daichu = 0.1 ** (len(s) - 1 - index_point)
            s = s[0:index_point] + s[index_point + 1:]  # 这里是除去小数点
        from functools import reduce
        result1 = reduce(lambda x, y: x * 10 + y, map(char2num, s))
        return result1 * daichu

    def __init__(self, idx, str):
        self.idx = idx
        l = str.split('_')
        # print(l)
        self.x = int(self.str2float(l[0]))
        self.y = int(self.str2float(l[1]))
        self.w = int(self.str2float(l[2]))
        self.h = int(self.str2float(l[3]))

    def draw_me(self, image):
        cv.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0))

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image, str(self.idx), (self.x + int(self.w / 2), self.y + int(self.h / 2)), font, 1, (0, 255, 0), 1,
                   cv.LINE_AA)

    def print_me(self):
        print("x,y, w, h: %d, %d, %d, %d" % (self.x, self.y, self.w, self.h))


class CreateCoco:

    def __init__(self):
        self.img_list = []
        self.annotation_id = 0

        self.coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

    def load_csv(self, path):
        with open(path) as csv_file:
            fields = ['name', 'objs']
            reader = csv.DictReader(csv_file, fieldnames=fields)
            for row in reader:
                tp_img = (row['name'], row['objs'])
                self.img_list.append(tp_img)

    def add_images(self):

        date_captured = datetime.datetime.utcnow().isoformat(' ')
        license_id = 1
        coco_url = ''
        flickr_url= ''

        for idx in range(len(self.img_list)):
            item = self.img_list[idx]
            image = PIL.Image.open("images//"+item[0])
            image_info = {
                "id": idx + 1,
                "file_name": item[0],
                "width": image.size[0],
                "height": image.size[1],
                "date_captured": date_captured,
                "license": license_id,
                "coco_url": coco_url,
                "flickr_url": flickr_url
            }
            self.coco_output["images"].append(image_info)

    def addAnnoItem(self, image_id, category_id, bbox):
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])

        annotation_item['segmentation'].append(seg)

        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        self.coco_output['annotations'].append(annotation_item)

    def create_coco(self):
        #add images
        self.add_images()
        #add annotations:
        for idx in range(len(self.img_list)):
            item = self.img_list[idx]
            l_objs = item[1].split(';')
            for i in range(len(l_objs)):
                rect = Rect(i, l_objs[i])
                bbox=[]
                bbox=[rect.x,rect.y, rect.w, rect.h]
                category_id = 3
                self.addAnnoItem(idx+1,category_id,bbox )
        with open('out.json', 'w') as output_json_file:
            json.dump(self.coco_output, output_json_file)


crt = CreateCoco()
crt.load_csv('book1.csv')
crt.create_coco()
