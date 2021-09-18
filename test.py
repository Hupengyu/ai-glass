import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from utils.datasets import LoadStreams
# from utils.plots_chinese import plot_one_box
import base64
import requests
from threading import Thread
# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from hash_rmrept import hash_rmrept
import cv2
from yolo3_x import YOLO
from PIL import Image
import numpy as np
import base64
import requests
from io import BytesIO
import torch.multiprocessing
from multiprocessing import Value
import time
import shutil
import datetime
from pathlib import Path
import glob
import threading
import torch.backends.cudnn as cudnn
from multi_rtmp_datasets import LoadStreams
pid_info_path = 'server/ProjectID.txt'
rtmpurl_pid_path = './server/c_camera.txt'
result_paths = 'result'
fixid = 'server/Fixed_ID.txt'
path_url = 'server/url_pids.txt'
yolo = YOLO()
alive = Value('b', False)
import os
from retinaface import Retinaface
retinaface = Retinaface()
def base_img(img_im):
    return base64.b64encode(cv2.imencode('.jpg',img_im)[1]).decode()
#去重
def image_rmrept(frame,pic_path):
    #reba = face_rmrept(frame, danger_list, face_path, pic_path)

    res = hash_rmrept(frame, pic_path)
    if res == 'same_pic':

        return False
    elif res == 'diff_pic':
        return True
def post(fras,data,dic,_img_path):
    with open(fixid, 'r') as f:
        fixs = f.readlines()
        for fix in fixs:
            if fix is None:
                break
            else:
                ffix = fix.split(',')
                ProjectID = ffix[0]
                BoxCode = ffix[1]
                backurl = ffix[2]
    if _img_path is not None:
        values = dic.get(_img_path)

        value = values.split(',')
    else:
        pass
    headers = {'Content-Type':'application/json;charset=UTF-8'}
    value = {"base64": base_img(fras),
             "checkdate": str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),
             "rtmpurl": value[-1],
             "data": data,
             "BoxCode": BoxCode,
             "CameraCode": _img_path,
             "ProjectID": ProjectID,

             }
    t2 = time.time()
    #print(value)
    #param = json.dumps(value,cls=MyEncoder,indent=4)
    #print(value)
    #try:
    res = requests.post(url=backurl,json=value,headers=headers)
    print(res.text)
    # except:
    #     res = "timeout"
    #     print(res)
# 每次程序启动，自动清空txt中数据的函数
def read_account(filename):
    with open(filename, 'r+', encoding='utf-8') as f:
        res = f.readlines()
        f.seek(0)
        f.truncate()
        f.close()
def get_url():
    with open(rtmpurl_pid_path, 'r') as f:
        dic = {}
        lines = f.readlines()
        for line in lines:
            if line is None:
                break
            else:
                re = line.strip('\n').split('_')
                # for line in lines:
                key = re[2]
                #value = str(re[0:3])

                CameraCode = re[1]
                ProjectID = re[0]
                url = re[-1]
                dic.setdefault(key, ProjectID+','+CameraCode+','+str(url))
        f.close()
    return dic
def thread_read_backurl():
    global rtmp_url,back_url
    while True:
        temp_rtmp_url = []
        temp_back_url = []
        with open(path_url, 'r', encoding='utf-8') as txt:
            lines = txt.readlines()
            length = len(lines)
            for i in range(length):
                if i % 2 == 0:
                    temp_rtmp_url.append(lines[i].strip())
                    if i+1 > length - 1:
                        break
                    temp_back_url.append(lines[i + 1].strip())
        txt.close()
        rtmp_url = temp_rtmp_url
        back_url = temp_back_url
        time.sleep(3)
def detect(save_img=False):

    # thread = Thread(target=thread_read_backurl, daemon=True)
    # thread.start()
    # time.sleep(1) #保证拿到的是更新后的rtmp_url和back_url

    # source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    source = 'server/url_pids.txt'


    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(  # isnumeric()方法检测字符串是否只由数字组成。
        ('rtsp://', 'rtmp://', 'http://'))


    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=640, stride=32)
    # else:
    #     dataset = LoadImages(source, img_size=640, stride=32)
    # traj = 0
    while True: #模型初始化检测，如果rtmp_url 为空，则暂时静置模型
        # if len(rtmp_url) > 0 and len(back_url) > 0:

        for img in dataset:
            print(img)
                # img = retinaface.detect_image(img)
            # image, detection = np.array(list(yolo.detect_image(img)), dtype=object)
            # print(image)
            # print(detection)
                # data = []
                # boxes = []
                # if len(detection):
                #     for i in range(len(detection)):
                #         boxes.append(detection[i][:4])
                #         detections = dict()
                #         type = {"type": detection[i][-1]}
                #         flag = {"flag": 3}
                #         pos = {"pos": ','.join(str(i) for i in (detection[i][:4]))}
                #
                #         detections.update(type)
                #         detections.update(flag)
                #         detections.update(pos)
                #         data.append(detections)
                #
                #     # new_time = time.strftime('%H%M%S', time.localtime())
                #
                #     # img_path_img = im_path.split('/')[-1]
                #     #
                #     name = '_' + "%02d" % (traj / 4) + '.jpg'  # 这里假设有四个rtmp流，但是在实际过程中，应该是 len(rtmp_url)
                #     traj += 1
                #     cv2.imwrite(result_paths + name, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
if __name__ == '__main__':
    detect()