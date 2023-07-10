import numpy as np
from PIL import Image
from PIL import ImageChops
import math
import operator
from functools import reduce
from airtest.aircv.cal_confidence import *
from SSIM_PIL import compare_ssim
from rembg import remove

#### hash
#转换为哈希码
def get_hash(srcimg, dstimg):
    print("当前图像的尺寸为：",dstimg.size)
    hh = srcimg.height
    ww = srcimg.width
    img = dstimg.resize((hh,ww), Image.ANTIALIAS).convert('L')#灰度处理

    print("转换图像尺寸为：",hh,ww)
    avg = sum(list(img.getdata())) /(hh*ww)  # 计算像素平均值
    s = ''.join(map(lambda i: '0' if i < avg else '1', img.getdata()))  # 每个像素进行比对,大于avg为1,反之为0
    print("一共有{}个像素点".format(len(s)))
    t=''.join(map(lambda j: '%x' % int(s[j:j+4], 2), range(0, hh*ww, 4)))#将二进制转换为16进制
    return t
# 进行hash码的逐位对比
def diff_hash(img1, img2):
    hash1= get_hash(img1, img1)
    hash2= get_hash(img1, img2)
    print("哈希值一共有{}位".format(len(hash1)))
    hh = img1.height
    ww = img1.width
    # >=98%的相似度才可以认为两个图片相同
    n = hh * ww // 200
    print("如果有{}个哈希值不同则认为两个图片不同".format(n))
    assert len(hash1) == len(hash2)
    sum = 0
    for ch1, ch2 in zip(hash1, hash2):
        if ch1 != ch2:
            sum += 1
    print("不同的哈希值有{}位".format(sum))
    # 比较两个hash值不同的位数，n为阈值，大于该阈值则认为两者是不同的图片
    zui = False if sum > n else True
    print(zui)


#### ssim
def compare_image_in_ssim(imgpath1, imgpath2):
  image1 = Image.open(imgpath1)
  image2 = Image.open(imgpath2)
  value = compare_ssim(image1, image2)
  return value

#### airtest compare
def airtest_compare(img1, img2):
    confidence= cal_ccoeff_confidence(img1, img2)
    return confidence

#### image chop
def compare_images(path_one, path_two, diff_save_location= None):
    image_one = Image.open(path_one)
    image_two = Image.open(path_two)
    diff = ImageChops.difference(image_one, image_two)
    if diff.getbbox() is None:
        # 图片间没有任何不同则直接退出
        print("我们是一模一样的图片")
    else:
        #diff.save(diff_save_location)
        print("not same")

##### histogram
# 将像素点数组中最大值设置为0
def set_max_to_zero(arr):
    max_val = np.max(arr)  # 找到数组中的最大值
    for i in range(len(arr)):
        if arr[i]== max_val:
            arr[i]= 0
    return arr

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

def image_contrast(img1, img2):
    # 转换为灰度图
    image1 = Image.open(img1).convert('L')
    image2 = Image.open(img2).convert('L')

    # 将目标图片转化大小与基准图片一致
    hh= image1.height
    ww= image1.width
    image2.resize((hh, ww))

    # 出现次数最多的像素点和出现次数
    mp1, mv1= most_value(image1)
    mp2, mv2= most_value(image2)

    # 参与比较的像素点所占比例
    r1= (hh*ww-mv1)/(hh*ww)
    r2= (hh*ww-mv2)/(hh*ww)

    # 获取图片像素直方图
    h1 = image1.histogram()
    h2 = image2.histogram()

    # 将像素值最多的作为背景设置为0
    h1= set_max_to_zero(h1)
    h2= set_max_to_zero(h2)

    # 进行度量
    result = math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))

    return result

def remove_bg(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)


if __name__ == "__main__":
    path1= 'dataset/gui_test/pic2.png'
    path2= 'dataset/gui_test/pos2.png'
    path3= 'dataset/gui_test/neg2.png'

    pic= Image.open(path1)
    pos= Image.open(path2)
    neg= Image.open(path3)

    # 去除背景函数
    # new_path1= 'dataset/gui_test/nbg_pic1.png'
    # #remove_bg(path1, new_path1)
    # nbg_pic1= Image.open(new_path1)

    # histogram
    # p= image_contrast(path1, path2)
    # n= image_contrast(path1, path3)

    # imagechops  不起作用
    # p= compare_images(path1, path2)
    # n= compare_images(path1, path3)

    #airtest  受分辨率影响
    # pic= np.array(pic)
    # pos= np.array(pos)
    # neg= np.array(neg)
    # p= airtest_compare(pic, pos)
    # n= airtest_compare(pic, neg)

    #ssim  需要分辨率一致才能对比!
    # p= compare_image_in_ssim(path1, path2)
    # n= compare_image_in_ssim(path1, path3)

    #hash
    diff_hash(pic, pos)
    diff_hash(pic, neg)


    # print(p)
    # print(n)