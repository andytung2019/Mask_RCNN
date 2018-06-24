import os, sys
import random
import csv
import shutil

LABEL_DIVIDE = 2000 


class DivideData:

    def __init__(self):
        self.img_list = []

    def load_csv(self, path):
        with open(path) as csv_file:
            fields = ['name', 'rois']
            reader = csv.DictReader(csv_file, fieldnames=fields)
            for row in reader:
                img = {'image_name': row['name'], 'rois': row['rois']}
                self.img_list.append(img)

    def divide_data(self):
        random.shuffle(self.img_list)
        self.move_data(self.img_list[:LABEL_DIVIDE])
        self.write_to_csv("valid_2k.csv", self.img_list[0:LABEL_DIVIDE])
        self.write_to_csv("train_8k.csv", self.img_list[LABEL_DIVIDE:])

    def write_to_csv(self, path, list_out):
        with open(path, 'a') as out_csv:
            fields = ['image_name', 'rois']
            for i in range(len(list_out)):
                r = list_out[i]
                writer = csv.DictWriter(out_csv, fieldnames=fields)
                writer.writerow({'image_name': r['image_name'], 'rois': r['rois']})

    def move_data(self, list_img):
        for item in list_img:
            shutil.move("train/" + item['image_name'], "valid/" + item['image_name'])


data = DivideData()
data.load_csv("train_1w.csv")
data.divide_data()
