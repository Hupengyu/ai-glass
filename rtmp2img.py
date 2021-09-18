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
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


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
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):

            if s not in camera_ip_set.keys():

                camera_ip_set.setdefault(s, "%03d" % (i + 1))
                if not os.path.exists(os.path.join(result_path, camera_ip_set[s])):
                    os.makedirs(os.path.join(result_path, camera_ip_set[s]), exist_ok=True)

            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap, s, camera_ip_set]), daemon=True)
            print(f' success ({w}x{h} at {fps:.2f} FPS).')
            thread.start()
        print('')  # newline

    def update(self, index, cap, s, camera_ip_set):
        # Read next stream frame in a daemon thread
        n = 0
        if not os.path.exists(os.path.join(result_paths, camera_ip_set[s])):
            os.makedirs(os.path.join(result_paths, camera_ip_set[s]))
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            new_time = time.strftime('%H%M%S', time.localtime())
            name = camera_ip_set[s] + '_' + new_time + '.jpg'
            path_a = os.path.join(result_path, camera_ip_set[s], name)
            endpath = os.path.join(result_paths, camera_ip_set[s], name)
            if n == 10:  # read every 4th frame
                success, im = cap.retrieve()
                try:
                    self.imgs[index] = im if success else self.imgs[index] * 0
                    n = 0
                    self.imgs[index].shape
                    cv2.imwrite(path_a, self.imgs[index])
                    shutil.move(path_a, endpath)
                except Exception as e:
                    print(e)
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        # img0 = self.imgs.copy()
        img0 = self.imgs.copy()
        for x in img0:
            #
            # # Letterbox
            # img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]
            # Stack
            img = np.stack(x, 0)
            # Convert
            # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416 608 608 *3

            img = np.ascontiguousarray(img)

        return img

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


if __name__ == '__main__':
    from yolo3_x import YOLO

    yolo = YOLO()
    import torch

    source = 'server/url_pids.txt'
    dataset = LoadStreams(source, img_size=608, stride=32)
    for image in dataset:
        pass
