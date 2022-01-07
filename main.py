import torch
from rtmp2video import LoadStreams
from yolo4 import YOLO
import subprocess as sp
from threading import Thread
import multiprocessing
import os
from tools import image_rmrept, post
import time

confi = False
status = False
url_init = []
rtsp_init = []
predict = dict()
images = {}
fixid = './model_data/BoxStatu.txt'
source = './model_data/url_cameras.txt'
yolo = YOLO()


def detection(dataset):
    global source, images, predict
    print('process detection success!')

    dataset = LoadStreams(dataset, img_size=608, stride=32)  # 加载txt中所有的视频流
    while True:

        imgdict = []
        i = 0
        for path, im0s in dataset:

            image_sets = im0s.copy()
            result = YOLO().detect_image(image_sets, im0s, path)    # 检测代码
            path_ = result.keys()

            for i, j in enumerate(path_):
                pre_ = result.get(j)[-1]
                img_ = result.get(j)[0]
                # img_re = retinaface.detect_image(img_)
                img_re = img_

                data_ = result.get(j)[1]
                print(data_)
                if len(data_):

                    if j not in predict.keys():

                        post(img_re, data_, j)
                        print(data_)
                        predict[j] = pre_
                        images[j] = img_re


                    else:
                        result_img = image_rmrept(images.get(j), img_re)
                        if pre_ == predict.get(j) and result_img:
                            predict[j] = pre_
                            images[j] = img_re
                            continue
                        else:

                            post(img_re, data_, j)
                            print(data_)
                            images[j] = img_re
                            predict[j] = pre_


def main():
    # 创建子进i程
    global path, url_init, status

    while True:

        with open(fixid, 'r', encoding='utf-8') as f_f:  # 打开盒子配置文件
            firs = f_f.readlines()
        with open(source, 'r', encoding='utf-8') as f_s:  # 打开视频流txt
            rtps = f_s.readlines()

        if len(firs) > 0 and len(rtps) > 0:
            urls_now = [x.strip('\n') for x in rtps]  # 预处理txt中url队列并合并成单个内容

            # 验证txt内容是否改变
            # txt内容不改变则一直验证
            if urls_init == urls_now:
                continue
            # txt内容变化则开新的进程
            else:
                urls_init = urls_now
                time.sleep(3)
                status = True

            if status:
                status = False
                ctx = multiprocessing.get_context('spawn')
                son_p1 = ctx.Process(target=detection, args=(source,), daemon=True)
                son_p1.start()
                print("开始检测")



        else:
            status = False


if __name__ == '__main__':
    main()
