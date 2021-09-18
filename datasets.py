# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
import multiprocessing
import signal
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm



# count_labels_logger = logging.getLogger('count_labels_logger')
#
# logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
#                     filename='count_labels_logger',
#                     filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#                     #a是追加模式，默认如果不写的话，就是追加模式
#                     format=
#                     '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#                     #日志格式
#                     )

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s





class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


result_paths = '../result'


camera_ip_set = {}
class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, url_path='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.threadId = []
        self.url_path = url_path

        if os.path.isfile(url_path):
            with open(url_path, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [url_path]

        n = len(sources)
        self.imgs = []

        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.sources.sort()
        sources.sort()

        self.stop_threads = False
        for i, s in enumerate(sources):
            if s not in camera_ip_set.keys():

                camera_ip_set.setdefault(s, "%03d" % i)
                if not os.path.exists(os.path.join(result_paths, camera_ip_set[s])):
                    os.makedirs(os.path.join(result_paths, camera_ip_set[s]), exist_ok=True)

            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)  # 当为0时，调用摄像头进行检测

            # assert cap.isOpened(), f'Failed to open {s}'
            if not cap.isOpened():
                n -= 1
                print("存在无效路径：", s, "将从文本中删除")
                with open(url_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # print(lines)
                with open(url_path, "w", encoding="utf-8") as f_w:
                    for line in lines:
                        if s in line:
                            continue
                        f_w.write(line)
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            # print("cap.read()",cap.read())
            # _, self.imgs[i] = cap.read()  # guarantee first frame
            self.imgs.append(cap.read()[1])
            # process = multiprocessing.Process(target=self.update, args=([len(self.imgs)-1, cap]), daemon=True)
            thread = Thread(target=self.update, args=([len(self.imgs)-1, cap]), daemon=True)
            print(f' success ({w}x{h} at {fps:.2f} FPS).')
            thread.start()
            self.threadId.append(thread)
            # process.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')


    def call(self, url_path='streams.txt'):
        if os.path.isfile(url_path):
            with open(url_path, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [url_path]
        sources.sort()

        sources1 = [clean_str(x) for x in sources]  # clean source names for later

        # 排序，保证顺序一致
        sources1.sort()


        # 杀死处理img获取的进程
        thresh_hold = 20
        biaozhi = False

        for cur_img in self.imgs:
            value = cur_img.sum(2).sum(1).sum(0)
            if value < thresh_hold:
                biaozhi = True
                break

        if sources1 != self.sources or biaozhi:
            self.stop_threads = True
            for t1 in self.threadId:
                t1.join()
                print('thread killed')

            self.threadId = []
            self.stop_threads = False
            self.sources = sources1 # 更新rtmp

            n = len(sources)
            self.imgs = []
            # self.sources = [clean_str(x) for x in sources]  # clean source names for later
            for i, s in enumerate(sources):
                # Start the thread to read frames from the video stream
                print(f'{i + 1}/{n}: {s}... ', end='')
                cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)  # 当为0时，调用摄像头进行检测
                # assert cap.isOpened(), f'Failed to open {s}'
                if not cap.isOpened():
                    n -= 1
                    print("存在无效路径：", s, "将从文本中删除")
                    with open(url_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        # print(lines)
                        f.close()
                    with open(url_path, "w", encoding="utf-8") as f_w:
                        for line in lines:
                            if s in line:
                                continue
                            f_w.write(line)
                        f_w.close()
                    continue

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) % 100
                # print("cap.read()",cap.read())
                # _, self.imgs[i] = cap.read()  # guarantee first frame
                self.imgs.append(cap.read()[1])
                # process = multiprocessing.Process(target=self.update, args=([len(self.imgs)-1, cap]), daemon=True)
                thread = Thread(target=self.update, args=([len(self.imgs) - 1, cap]), daemon=True)
                print(f' success ({w}x{h} at {fps:.2f} FPS).')
                thread.start()
                self.threadId.append(thread)
                # process.start()
            print('')  # newline
            time.sleep(0.1) # 休眠0.1，保证线程已开始工作

            # check for common shapes
            if len(self.imgs)>0:
                s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
                self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
            else:
                x = np.zeros((576,1024,3),dtype='uint8')
                self.imgs.append(x)
                self.sources.append("null_path")
                s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
                self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

            if not self.rect:
                print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # self.processID.append(os.getpid())
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()  #从视频文件或捕获设备中抓取下一帧。
            if n == 20:  # read every 4th frame
                success, im = cap.retrieve() # 解码并返回抓取的视频帧。
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(0.01)  # wait time
            if self.stop_threads:
                break

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.call(self.url_path)
        self.count += 1
        img0 = self.imgs.copy()
        for x in img0:

        #
        # # Letterbox
        #img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]
        # Stack
            img = np.stack(x, 0)
            # Convert
            #img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416 608 608 *3

            img = np.ascontiguousarray(img)



        return  self.sources,img

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]




# Ancillary functions --------------------------------------------------------------------------------------------------




def letterbox(img, new_shape=(608, 608), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

if __name__ == '__main__':
    from yolo3_x import YOLO
    yolo = YOLO()
    import torch
    source =r'K:\JKY-AI-CX\server\url_pids.txt'
    dataset = LoadStreams(source, img_size=608, stride=32)
    for path,image in dataset:
        print("路径：", path, "文件：",image)
        #print(path+image)
       # print(detection)