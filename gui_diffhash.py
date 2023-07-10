import argparse
import os, sys
import torchvision.models as models
import torchvision.transforms
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
sys.path.append('./')
import os.path as osp
import torch
from PIL import Image
import random
from collate import SimCLRCollateFunction
from Trainer import trainer_baseline_ce
# from data_clus_for_dividemix import  get_train_loader
# from data_clus import office_load_idx
# import clustering
from network import Model
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from utils import *
from PIL import ImageGrab
from scipy import spatial

#计算图片像素值的众数
def most_value(img):
    # 将图片转换为 NumPy 数组
    image_array = np.array(img)

    # 统计像素点值和对应的频次
    unique_values, counts = np.unique(image_array, return_counts=True)

    # 找到出现次数最多的像素点值及其频次
    most_common_index = np.argmax(counts)
    most_common_pixel_value = unique_values[most_common_index]
    most_common_count = counts[most_common_index]

    # print("相同像素点值最多的是:", most_common_pixel_value)
    # print("它出现的次数:", most_common_count)
    return most_common_pixel_value, most_common_count


# 计算的结果为相似度 值越大越相似

def load_img(path):
    img= Image.open(path)
    img= img.convert('RGB')
    return img

# 正则化图像
def regularizeImage(img, size = (9, 8)):
    return img.resize(size).convert('L')

## 差异哈希
# 计算hash值
def getHashCode(img, size=(9, 8)):
    result = []
    for i in range(size[0] - 1):
        for j in range(size[1]):
            current_val = img.getpixel((i, j))
            next_val = img.getpixel((i + 1, j))
            if current_val > next_val:
                result.append(1)
            else:
                result.append(0)

    return result

# 比较哈希值
def compHashCode(hc1, hc2):
    cnt = 0
    for i, j in zip(hc1, hc2):
        if i == j:
            cnt += 1
    return cnt

# 计算差异哈希算法相似度
def caldHashSimilarity(img1, img2):
    img1 = regularizeImage(img1)
    img2 = regularizeImage(img2)
    hc1 = getHashCode(img1)
    hc2 = getHashCode(img2)
    return compHashCode(hc1, hc2)


if __name__ == "__main__":
    path1= 'dataset/gui_test/pic3.png'
    path2= 'dataset/gui_test/pos3.png'
    path3= 'dataset/gui_test/neg3.png'

    pic= Image.open(path1).convert('L')
    pos= Image.open(path2).convert('L')
    neg= Image.open(path3).convert('L')

    # resize正负样本
    hh= pic.height
    ww= pic.width
    pos= pos.resize((hh, ww))
    neg= neg.resize((hh, ww))

    # p= caldHashSimilarity(pic, pos)
    # q= caldHashSimilarity(pic, neg)
    #
    # print(p, q)

    print(most_value(neg))






