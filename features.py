
# 提取图片的特征
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import os
import numpy as np

# import resnet50
resnet50 = models.resnet50(pretrained=True, progress=True)
# print(resnet50)

# 定义图片处理器
tf = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.Resize((360,360)),
    transforms.ToTensor()
])

# 定义图片路径
image_dir = './01'
# 定义导入的图片向量

for root, dirs, files in os.walk(image_dir):
    files.sort()
    i = 0
    # 定义所有数据的保存向量
    feature = torch.tensor([])
    for f in files:
        image_path = os.path.join(root, f)
        # print(image_path)
        print(i)
        i += 1
        image_tmp = tf(image_path).unsqueeze(0)
        feature_tmp = resnet50(image_tmp)
        feature = torch.cat((feature, feature_tmp), 0)

# 将所有的图像过网络提取特征
# feature = resnet50(image)
feature = feature.detach().numpy()
print(feature.shape)

np.save('feature.npy',feature)


