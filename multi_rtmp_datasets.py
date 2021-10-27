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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
import re


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, url_path='server/url_cameras.txt', img_size=608, stride=32):
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
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years