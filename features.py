
# 提取图片的特征
import torch
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
    batch_size = 2
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=32)


    #  iterator = iter(train_loader)
    #  #  x is a list which contains [I0, I1, I2, I3, I4]
    #  x = next(iterator)
    #  # stack the different image
    #  x = torch.cat([x[i] for i in range(5)], dim=0)
    #  print(len(x))
    #
    #  # import resnet50 model
    #  resnet50 = models.resnet50(pretrained=True, progress=True)
    #  resnet50.eval()
    #  image_feature = resnet50(x) # this size is [2,1000]
    #  print(x.shape)
    #  print(image_feature.shape)
    

    print(device)
    print('the end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # prepare net 
    # TODO
    #  model =
    

if __name__ == "__main__":
    main()
