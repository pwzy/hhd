
# 提取图片的特征
import torch
from torch import device
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from dataset import *

def main():
    # prepaer data``
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_set = Data_Set('../data/ped2/training/frames', resize_height=360, resize_width=360, time_steps=5)
    train_loader = DataLoader(data_set, batch_size=2, shuffle=True, num_workers=32)


    #  iterator = iter(train_loader)
    #  x = next(iterator)
    #  resnet50 = models.resnet50(pretrained=True, progress=True)
    #  resnet50.eval()
    #  image_feature = resnet50(x[0]) # this size is [2,1000]
    print('test')

    # prepare net 
    # TODO
    #  model =
    

if __name__ == "__main__":
    main()
