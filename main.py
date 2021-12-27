import cv2
import os
import torch

from Gen_train_data import Gen_VOC_data
from model import Model
from Loss import loss
from VOCDataSet import VOCDataset

label_file = 'label_file.txt'
# voc_path = '/media/li/b806bc78-4cbd-4e31-ba5c-e0212d292e732/data/VOC/VOC2012'
# VOC_data = Gen_VOC_data()
# VOC_data.data_process(voc_path,label_file)
train_dataset = VOCDataset(label_file)


use_gpu = torch.cuda.is_available()
assert use_gpu, 'Current implementation does not support CPU mode. Enable CUDA.'
print('CUDA current_device: {}'.format(torch.cuda.current_device()))
print('CUDA device_count: {}'.format(torch.cuda.device_count()))

# 重写yolov1 模型
yolov1_model = Model()

# Training hyper parameters.
init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 64

# 调整学习率
criterion = loss()
optimizer = torch.optim.SGD(yolov1_model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

#