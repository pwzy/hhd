
# 提取图片的特征
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

# import resnet50
resnet50 = models.resnet50(pretrained=True, progress=True)
# print(resnet50)


resnet50.eval()

