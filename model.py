import torch
import torch.nn as nn

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.block1 = self.conv_bolck(3,64,kernel_size=7,stride=2,padding=3,max_min_label=True)
        self.block2 = self.conv_bolck(64,192,kernel_size=3,padding=1,max_min_label=True)
        self.block3 = self.conv_bolck(192,128,kernel_size=1,padding= 0,max_min_label=False)
        self.block4 = self.conv_bolck(128,256,kernel_size=3,padding=1,max_min_label=False)
        self.block5 = self.conv_bolck(256,256,kernel_size=1,padding=0,max_min_label=False)
        self.block6 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=True)

        self.block7 = self.conv_bolck(512,256,kernel_size=1,padding=0,max_min_label=False)
        self.block8 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=False)
        self.block9 = self.conv_bolck(512,256,kernel_size=1,padding=0,max_min_label=False)
        self.bolck10 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=False)
        self.block11 = self.conv_bolck(512,256,kernel_size=1,padding=0,max_min_label=False)
        self.block12 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=False)
        self.block13 = self.conv_bolck(512,256,kernel_size=1,padding=0,max_min_label=False)
        self.block14 = self.conv_bolck(256,512,kernel_size=3,padding=1,max_min_label=False)
        self.block15 = self.conv_bolck(512,512,kernel_size=1,padding=0,max_min_label=False)
        self.block16 = self.conv_bolck(512,1024,kernel_size=3,padding=1,max_min_label=True)

        self.block17 = self.conv_bolck(1204,512,kernel_size=1,padding=0,max_min_label=False)
        self.block18 = self.conv_bolck(512,1024,kernel_size=3,padding=1,max_min_label=False)
        self.block19 = self.conv_bolck(1024,512,kernel_size=1,padding=0,max_min_label=False)
        self.block20 = self.conv_bolck(512,1024,kernel_size=3,padding=1,max_min_label=False)

        self.fc = self.fc_blck(1024,1000)
    def fc_blck(self,input,output):
        fc = nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(input, output)
        )
        return fc

    def conv_bolck(self,input,output,kernel_size,stride,padding,max_min_label):
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
        out = self.bolck1(x)
        out = self.bolck2(out)
        out = self.bolck3(out)
        out = self.bolck4(out)
        out = self.bolck5(out)
        out = self.bolck6(out)
        out = self.bolck7(out)
        out = self.bolck8(out)
        out = self.bolck9(out)
        out = self.bolck10(out)
        out = self.bolck11(out)
        out = self.bolck12(out)
        out = self.bolck13(out)
        out = self.bolck14(out)
        out = self.bolck15(out)
        out = self.bolck16(out)
        out = self.bolck17(out)
        out = self.bolck18(out)
        out = self.bolck19(out)
        out = self.bolck20(out)

        out = self.fc_blck(out)
