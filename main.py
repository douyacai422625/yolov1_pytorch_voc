import cv2
import os
import torch
from torch.utils.data import DataLoader
from Gen_train_data import Gen_VOC_data
from model import Model
from Loss import loss
from VOCDataSet import VOCDataset
import tqdm
from torch.autograd import Variable
from datetime import datetime
from tensorboardX import SummaryWriter

### 参数
# 超参数
init_lr = 0.001
momentum = 0.9
weight_decay = 5.0e-4
num_epoch = 300
batch_size = 1

print_freq = 5  # epoch 每过5次打印一次
tb_log_freq = 5 # epoch 第迭代5次写日志一次

label_file = 'label_file.txt'
# voc_path = '/home/li/Documents/data/VOC/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
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

# Open TensorBoardX summary writer
log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('yolo', log_dir)
writer = SummaryWriter(log_dir=log_dir)


for epoch in range(num_epoch):
    # Training.
    yolov1_model.train()
    total_loss = 0.0
    total_batch = 0

    # 训练
    for imgs, targets in train_loader:
        # Update learning rate.
        loss.update_lr(optimizer, epoch, float(epoch) / float(len(train_loader) - 1))
        lr =  loss.get_lr(optimizer)

        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        imgs, targets = imgs.cuda(), targets.cuda()

        # Forward to compute loss.
        preds = yolov1_model(imgs)
        LOSS = criterion(preds, targets)
        loss_this_iter = loss.item()
        total_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter

        # Backward to update model weight.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print current loss.
        if epoch % print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
            % (epoch, num_epoch, epoch, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))

        # TensorBoard.
        n_iter = epoch * len(train_loader) + epoch
        if n_iter % tb_log_freq == 0:
            writer.add_scalar('train/loss', loss_this_iter, n_iter)
            writer.add_scalar('lr', lr, n_iter)