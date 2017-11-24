import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names.append('siamnet')


class Network(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        def spp(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
            for i in range(len(out_pool_size)):
                # print(previous_conv_size)
                h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
                w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
                h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
                w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
                maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
                x = maxpool(previous_conv)
                if(i == 0):
                    spp = x.view(num_sample,-1)
                    # print("spp size:",spp.size())
                else:
                    # print("size:",spp.size())
                    spp = torch.cat((spp,x.view(num_sample,-1)), 1)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  16, 2), 
            conv_dw( 16,  32, 1),
            conv_dw( 32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            #nn.AvgPool2d(7),
            spp(x,1,[int(x.size(2)),int(x.size(3))],21)
        )
        self.fc = nn.Linear(512, 7)

    def forward_once(self, x):
        output = self.model(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2