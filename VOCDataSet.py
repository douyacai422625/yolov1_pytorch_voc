import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import random
import numpy as np

class VOCDataset(Dataset):
    def __init__(self,label_txt,train_label,img_size = 448,grid_size = 7,num_bboxes = 2,num_classes = 20):
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        self.train_label = train_label

        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        with open(label_txt) as f:
            lines = f.readlines()

        self.paths, self.boxes, self.labels = [], [], []
        for line in lines:
            box, label = [], []
            splitted = line.strip().split(' ')
            path = splitted[0]
            self.paths.append(path)
            for bbox_class in splitted[1:]:
                split_bbox_class = bbox_class.split(',')
                x1 = float(split_bbox_class[0])
                y1 = float(split_bbox_class[1])
                x2 = float(split_bbox_class[2])
                y2 = float(split_bbox_class[3])
                c =  int(split_bbox_class[4])
                box.append([x1, y1, x2, y2])
                label.append(c)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        boxes = self.boxes[idx].clone() # [n, 4]
        labels = self.labels[idx].clone() # [n,]
        for bbox in boxes:
            x1,y1,x2,y2 = bbox
            x1 = int(x1.numpy())
            y1 = int(y1.numpy())
            x2 = int(x2.numpy())
            y2 = int(y2.numpy())
        if self.train_label:
            new_img,boxes = self.random_flip(img,boxes)
            for bbox in boxes:
                x1, y1, x2, y2 = bbox
                x1 = int(x1.numpy())
                y1 = int(y1.numpy())
                x2 = int(x2.numpy())
                y2 = int(y2.numpy())
                try:
                    cv2.rectangle(new_img, pt1= (x1, y1), pt2 = (x2, y2), color=(0, 0, 255), thickness=2)
                except:
                    print(idx)

    def random_flip(self, img, boxes):
        # if random.random() < 0.5:
        #     return img, boxes

        h, w, _ = img.shape

        img = np.fliplr(img)

        x1, x2 = boxes[:, 0], boxes[:, 2]
        x1_new = w - x2
        x2_new = w - x1
        boxes[:, 0], boxes[:, 2] = x1_new, x2_new

        return img, boxes

    def __len__(self):
        return self.num_samples