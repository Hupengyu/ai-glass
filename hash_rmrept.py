import cv2
import os
import time
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib



matplotlib.use('Agg')
img_path = "imgg/001"

def pHash(img):
    img = cv2.resize(img, (32, 32))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

 
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree
 
 
def classify_hist_with_split(image1, image2, size=(256, 256)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data
 
 
def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def getImageByUrl(url):
    # 根据图片url 获取图片对象
    html = requests.get(url, verify=False)
    image = Image.open(BytesIO(html.content))
    return image


def bytes_to_cvimage(filebytes):
    image = Image.open(filebytes)
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img



def runtwoImageSimilaryFun(para1, para2):

    img1 = para1
    img2 = para2

    hash1 = pHash(img1)
    hash2 = pHash(img2)
    n3 = cmpHash(hash1, hash2)
    n4 = classify_hist_with_split(img1, img2)
    return n3, n4

def get_img_list(dir, firelist, ext=None):
    if os.path.isfile(dir):  # 如果是文�?
        if ext is None:
            firelist.append(dir)
        elif ext in dir[-3:]:
            firelist.append(dir)
    elif os.path.isdir(dir):  # 如果是目�?

        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            get_img_list(newdir, firelist, ext)
    return firelist


def read_img(image_path, fra):
    """
    Args:
        image_path:
        fra:

    Returns:    diff_image
                same_image
    """
    img_list = []
    diff_image = 'diff_image'
    same_image = 'same_image'
    fps_num = 0
    img_num = 0
    save_img = None
    imglist = get_img_list(image_path, [], 'jpg')

    imglist.sort()


    img = cv2.imread(imglist[-1], cv2.IMREAD_COLOR)

    n3, n4= runtwoImageSimilaryFun(fra, img)


    img_num += 1
    if n3 < 15 or n4 > 0.8:
        save_img = None
    else:
        fps_num += 1
        save_img = fra
        save_img = Image.fromarray(save_img)

    if save_img != None and fps_num == img_num:

        return diff_image
    return same_image


def hash_rmrept(frame, img_path):
    """
    Args:
        frame:
        img_path:

    Returns:    diff_pic
                same_pic
    """
    diff_pic = 'diff_pic'
    same_pic = 'same_pic'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if not os.listdir(img_path):
        return diff_pic

    #cv2.imshow('src', frame)

    # new_time = time.strftime('%H%M%S', time.localtime())
    # img_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    # # 设定时间，每天固定时刻清空已有文件夹中的图片
    # if new_time == str('235959'):
    #     for file in os.listdir(img_path):
    #         path_file = os.path.join(img_path, file)
    #         if os.path.isfile(path_file):
    #             os.remove(path_file)
    #             cv2.imwrite(img_path + '/' + str(pid)+"_"+ img_time +".jpg", frame)
    result = read_img(img_path, frame)

    if result == 'diff_image':

        return diff_pic
    elif result == 'same_image':
        return same_pic


if __name__ == "__main__":
    #
    cap = cv2.VideoCapture('pic_img/001/002_164438.jpg')
    while True:
        ret, fra_ = cap.read()
        if ret:
            print(hash_rmrept(fra_, img_path))

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('save.jpg', fra_)
    # print(new_time)
    #
    # def get_img_list(dir, firelist, ext=None):
    #     if os.path.isfile(dir):  # 如果是文�?
    #         print(len(dir))
    #         if ext is None:
    #             firelist.append(dir)
    #         elif ext in dir[-3:]:
    #             firelist.append(dir)
    #     elif os.path.isdir(dir):  # 如果是目�?
    #
    #         for s in os.listdir(dir):
    #             newdir = os.path.join(dir, s)
    #             get_img_list(newdir, firelist, ext)
    #     return firelist
    #


  #  imglist = get_img_list(image_path, [], 'jpg')

    import glob

    #imagelist = sorted(glob.glob(image_path))  # 读取带有相同关键字的图片名字，比上一中方法好
    #imagelist = cv2.imread(image_path,1)
    # imglist = get_img_list(image_path, [], 'jpg')
    # imglist.sort()
    # print(imglist)
    # print(imglist[-1])
