import glob
import logging
import math
import os
import random
import shutil
import time
from threading import Thread
import re
import cv2
import numpy as np

result_paths = 'result'
camera_ip_set = {}
result_path = 'results'
logger = logging.getLogger(__name__)


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="", string=s)


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


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=608, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]
        box = []

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [x for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            s = s.split(',')[0]
            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f'success ({w}x{h} at {fps:.2f} FPS).')
            thread.start()
            box.append([w, h, fps])
        # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        while cap.isOpened():
            # _, self.imgs[index] = cap.read()
            cap.grab()
            # read every 4th frame
            success, im = cap.retrieve()
            self.imgs[index] = im if success else self.imgs[index] * 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):

        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        # img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]
        #
        # # Stack
        # img = np.stack(img, 0)
        #
        # # Convert
        # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        # img = np.ascontiguousarray(img)

        return self.sources, img0

    def __len__(self):
        return 0


#
# rta = False
# rtmp_url = []
# def thread_read_backurl():
#     global rtmp_url,rta
#
#     while True:
#
#         with open('model_data/url_cameras.txt', 'r', encoding='utf-8') as txt:
#             lines = txt.readlines()
#
#             length = len(lines)
#             leng = len(rtmp_url)
#             txt.close()
#             if leng == 0:
#                 for i in range(length):
#                     rtmp_url.append(lines[i])
#                 rta = True
#
#             else:
#                 if lines == rtmp_url:
#                     continue
#                 else:
#                     rtmp_url.clear()
#                     rta = False
#                     time.sleep(3)
#         dataset = LoadStreams(rtmp_url, img_size=608, stride=32)
#         print(dataset)
#

if __name__ == '__main__':
    dir_l = './model_data/url_cameras.txt'

    dataset = LoadStreams(dir_l, img_size=640, stride=32)
    for path, im0s in dataset:
        image_sets = im0s.copy()
        imh = image_sets[0]
        imf = image_sets[1]
        cv2.imshow("video", imh)
        cv2.imshow("video2", imf)
