from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

import numpy as np

class VOCDataset(Dataset):
    def __init__(self,label_txt,img_size = 448,grid_size = 7,num_bboxes = 2,num_classes = 20):
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        with open(label_txt) as f:
            lines = f.readlines()

        box, label = [], []
        self.paths, self.boxes, self.labels = [], [], []
        for line in lines:
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
            a = 1
        self.num_samples = len(self.paths)


            # fname = splitted[0]
            # path = os.path.join(image_dir, fname)
            # self.paths.append(path)
            #
            # num_boxes = (len(splitted) - 1) // 5
            # box, label = [], []
            # for i in range(num_boxes):
            #     x1 = float(splitted[5*i + 1])
            #     y1 = float(splitted[5*i + 2])
            #     x2 = float(splitted[5*i + 3])
            #     y2 = float(splitted[5*i + 4])
            #     c  =   int(splitted[5*i + 5])
            #     box.append([x1, y1, x2, y2])
            #     label.append(c)
            # self.boxes.append(torch.Tensor(box))
            # self.labels.append(torch.LongTensor(label))

