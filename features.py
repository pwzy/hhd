
# 提取图片的特征
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from dataset import *
from gcn_model import GCN_Module

def main():

    batch_size = 2
    image_num = 5

    # prepaer data``
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_set = Data_Set('../data/ped2/training/frames', resize_height=360, resize_width=360, time_steps=5)
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=32)
    
    # import backbone model
    model_backbone = models.resnet50(pretrained=True, progress=True)
    model_backbone = model_backbone.to(device)
    # import gcn model 
    model_gcn = GCN_Module()
    model_gcn = model_gcn.to(device)  
    
    for epoch in range(10):
        for batchidx, image in enumerate(train_loader, 0):

            # 进行图像的堆叠，一次过网络，5个[batch,3,360,360] => [batch*5,3,360,360]
            image = torch.cat([image[i].to(device) for i in range(image_num)], dim=0) 
            #  print(image.shape)
            # 获得图像特征 大小为[batch*5, 1000]
            image_features = model_backbone(image)  # [10,3,360,360]
            #  print(image_features.shape)
            # 将每一张图片的特征分开，形成列表 image_features = [[batch_size,1000],[batch_size,1000],[batch_size,1000],...5次]
            image_features = [image_features[i*batch_size : i*batch_size+1] for i in range(image_num)]
            #  print(len(image_features))
            # 遍历batch
            for batch_clip in range(batch_size):
                I0 = image_features[0][batch_clip].unsqueeze(0)
                I1 = image_features[1][batch_clip].unsqueeze(0)
                I2 = image_features[2][batch_clip].unsqueeze(0)
                I3 = image_features[3][batch_clip].unsqueeze(0)
                I4 = image_features[4][batch_clip].unsqueeze(0)
                # I.shape is [1, 5, 1000]  
                I = torch.stack([I0, I1, I2, I3, I4], dim=1)

                print("data prepare done~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

                output_features, relation_graph = model_gcn(I)



    print(device)
    print('the end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # prepare net 
    # TODO
    #  model =
    

if __name__ == "__main__":
    main()
