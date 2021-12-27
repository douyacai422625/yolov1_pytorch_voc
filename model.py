import torch
import torch.nn as nn

class Flaten(nn.Module):
    def __init__(self):
        super(Flaten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self,num_bboxes = 2,num_classes = 20):
        super(Model, self).__init__()

        self.feature_size = 7
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.block1 = self.conv_bolck(3,64,kernel_size=7,stride=2,padding=3,max_min_label=True)

        self.block2 = self.conv_bolck(64,192,kernel_size=3,padding=1,max_min_label=True)

        self.block3 = self.conv_bolck(192,128,kernel_size=1,padding= 0,max_min_label=False)
        self.block4 = self.conv_bolck(128,256,kernel_size=3,padding=1,max_min_label=False)
        self.block5 = self.conv_bolck(256,256,kernel_size=1,padding=0,max_min_label=False)
        self.block6 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=True)

        self.block7 = self.conv_bolck(512,256,kernel_size=1,padding=0,max_min_label=False)
        self.block8 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=False)
        self.block9 = self.conv_bolck(512,256,kernel_size=1,padding=0,max_min_label=False)
        self.block10 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=False)
        self.block11 = self.conv_bolck(512,256,kernel_size=1,padding=0,max_min_label=False)
        self.block12 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=False)
        self.block13 = self.conv_bolck(512,256,kernel_size=1,padding=0,max_min_label=False)
        self.block14 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=False)
        self.block15 = self.conv_bolck(512,512,kernel_size=1,padding=0,max_min_label=False)
        self.block16 = self.conv_bolck(512,1024,kernel_size=3,padding=1,max_min_label=True)

        self.block17 = self.conv_bolck(1024,512,kernel_size=1,padding=0,max_min_label=False)
        self.block18 = self.conv_bolck(512,1024,kernel_size=3,padding=1,max_min_label=False)
        self.block19 = self.conv_bolck(1024,512,kernel_size=1,padding=0,max_min_label=False)
        self.block20 = self.conv_bolck(512,1024,kernel_size=3,padding=1,max_min_label=False)
        self.block21 = self.conv_bolck(1024,1024,kernel_size=3,padding=1,max_min_label=False)
        self.block22 = self.conv_bolck(1024,1024,kernel_size=3,stride = 2,padding=1,max_min_label=False)

        self.block23 = self.conv_bolck(1024,1024,kernel_size=3,padding=1,max_min_label=False)
        self.block24 = self.conv_bolck(1024,1024,kernel_size=3,padding=1,max_min_label=False)

        self.fc = self.fc_blck(1024,1000)

    def fc_blck(self,input,output):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes
        fc = nn.Sequential(
            Flaten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False), # is it okay to use Dropout with BatchNorm?
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )
        return fc

    def conv_bolck(self,input,output,kernel_size,padding,max_min_label,stride=1):
        if max_min_label:
            block = nn.Sequential(
                nn.Conv2d(input,output,kernel_size=kernel_size,stride=stride,padding = padding),
                nn.BatchNorm2d(output),
                nn.LeakyReLU(0.1,inplace=True),
                nn.MaxPool2d(2)
            )
        else:
            block = nn.Sequential(
                nn.Conv2d(input,output,kernel_size=kernel_size,stride=stride,padding = padding),
                nn.BatchNorm2d(output),
                nn.LeakyReLU(0.1,inplace=True)
            )
        return block

    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.block11(out)
        out = self.block12(out)
        out = self.block13(out)
        out = self.block14(out)
        out = self.block15(out)
        out = self.block16(out)
        out = self.block17(out)
        out = self.block18(out)
        out = self.block19(out)
        out = self.block20(out)
        out = self.block21(out)
        out = self.block22(out)
        out = self.block23(out)
        out = self.block24(out)

        out = self.fc(out)

        out = out.view(-1, self.feature_size, self.feature_size, 5 * self.num_bboxes + self.num_classes)
        print('final_model_shape : ',out.shape)
