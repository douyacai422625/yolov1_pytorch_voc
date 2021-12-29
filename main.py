import cv2
import os
import torch
from torch.utils.data import DataLoader
from Gen_train_data import Gen_VOC_data
from model import Model
from Loss import loss
from VOCDataSet import VOCDataset
import tqdm

### 参数
# 超参数
init_lr = 0.001
momentum = 0.9
weight_decay = 5.0e-4

batch_size = 1

label_file = 'label_file.txt'
# voc_path = '/media/li/b806bc78-4cbd-4e31-ba5c-e0212d292e732/data/VOC/VOC2012'
# VOC_data = Gen_VOC_data()
# VOC_data.data_process(voc_path,label_file)
train_dataset = VOCDataset(label_file,True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

use_gpu = torch.cuda.is_available()

# 重写yolov1 模型
yolov1_model = Model()

# 调整学习率
criterion = loss()
optimizer = torch.optim.SGD(yolov1_model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)




# 训练
for imgs, targets in train_loader:
    print('循环')
    a = 1