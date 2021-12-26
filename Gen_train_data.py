import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm, trange

class Gen_VOC_data:
    def __init__(self):
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

    def data_process(self,path,label_file):
        JPEGImage = os.path.join(path,'JPEGImages')
        Annotations = os.path.join(path,'Annotations')
        img_list = os.listdir(JPEGImage)

        train_file = open(label_file,'w')
        pbar = tqdm(img_list)
        for img_name in pbar:
            pbar.set_description('VOC process:')

            img_path = os.path.join(JPEGImage,img_name)
            anno_name = img_name.split('.')[0] + '.xml'
            anno_path = os.path.join(Annotations,anno_name)
            tree = ET.ElementTree(file=anno_path)
            root = tree.getroot()

            train_file.write(img_path)

            for obj in root.iter('object'):
                obj_name = obj.find('name').text
                difficult = 0
                if obj.find('difficult') != None:
                    difficult = obj.find('difficult').text
                if int(difficult) == 1 or not obj_name in self.classes:
                    continue

                obj_id = self.classes.index(obj_name)
                xmlbox = obj.find('bndbox')
                b = [xmlbox.find(value).text for value in ('xmin', 'ymin', 'xmax', 'ymax')]
                content = ','.join([str(x) for x in b])
                train_file.write(' ' + content+',' + str(obj_id))
            train_file.write('\n')
        train_file.close()