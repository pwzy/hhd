
# 提取图片的特征
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from dataset import *

def main():

    batch_size = 2
    image_num = 5

    # prepaer data``
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_set = Data_Set('../data/ped2/training/frames', resize_height=360, resize_width=360, time_steps=5)
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=32)

    model_backbone = models.resnet50(pretrained=True, progress=True)
    
    for epoch in range(10):
        for batchidx, image in enumerate(train_loader, 0):

            # 进行图像的堆叠，一次过网络，5个[batch,3,360,360] => [batch*5,3,360,360]
            image = torch.cat([image[i] for i in range(image_num)], dim=0) 
            #  print(image.shape)
            # 获得图像特征 大小为[batch*5, 1000]
            image_features = model_backbone(image) 
            #  print(image_features.shape)
            # 将每一张图片的特征分开，形成列表 image_features = [[batch_size,000],[batch_size,1000],[batch_size,1000],...5次]
            image_features = [image_features[i*batch_size : i*batch_size+1] for i in range(image_num)]
            print(len(image_features))




        

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
