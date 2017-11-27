import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataset import *

class PLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=10.0):
        super(PLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        total = output1+output2
        trans_distance = F.pairwise_distance(total[:4], label[:4])
        rot_distance = F.pairwise_distance(total[4:], label[4:])
        loss = trans_distance + self.margin*rot_distance
        return loss