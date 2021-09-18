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
import glob

pid_info_path = 'server/ProjectID.txt'
rtmpurl_pid_path = './server/c_camera.txt'
result_paths = 'result'
fixid = 'server/Fixed_ID.txt'
yolo = YOLO()
alive = Value('b', False)
import os
from retinaface import Retinaface

retinaface = Retinaface()


def frame2base64(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)  # 将每一帧转为Image
    # cv2.imwrite(image_path + '/'  + "1.jpg", frame)
    output_buffer = BytesIO()  # 创建一个BytesIO
    img.save(output_buffer, format='JPEG')  # 写入output_buffer
    byte_data = output_buffer.getvalue()  # 在内存中读取
    Base64 = base64.b64encode(byte_data)  # 转为BASE64
    return Base64


def base_img(img_im):
    return base64.b64encode(cv2.imencode('.jpg', img_im)[1]).decode()


# 去重
def image_rmrept(frame, pic_path):
    # reba = face_rmrept(frame, danger_list, face_path, pic_path)

    res = hash_rmrept(frame, pic_path)
    if res == 'same_pic':

        return False
    elif res == 'diff_pic':
        return True


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
                # value = str(re[0:3])

                CameraCode = re[1]
                ProjectID = re[0]
                url = re[-1]
                dic.setdefault(key, ProjectID + ',' + CameraCode + ',' + str(url))
        f.close()
    return dic


def post(fras, data, dic, _img_path):
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
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    value = {"base64": base_img(fras),
             "checkdate": str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),
             "rtmpurl": value[-1],
             "data": data,
             "BoxCode": BoxCode,
             "CameraCode": _img_path,
             "ProjectID": ProjectID,

             }
    t2 = time.time()
    # print(value)
    # param = json.dumps(value,cls=MyEncoder,indent=4)
    # print(value)
    # try:
    res = requests.post(url=backurl, json=value, headers=headers)
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


def manage_dict(thresh_hold, pid):  # 处理字典超时方法
    global yolo, folder_dict
    delete = []
    current_time = time.time()
    for key, val in list(folder_dict.items()):
        if (current_time - folder_dict[key]) >= thresh_hold:
            delete.append(key)  # 删除超时文件夹
            print('{}文件夹超时:'.format(key), (current_time - folder_dict[key]), )
    for i in range(len(delete)):
        folder_dict.pop(delete[i])
        lines = [l for l in open(pid_info_path, "r") if l.find(str(delete[i]) + ',' + str(pid)) != 0]

        with open(pid_info_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            f.close()


def to_work():
    #  global folder_dict
    global yolo, folder_dict
    dir_set = set()
    images_set = set()
    max_number = 5
    pid = os.getpid()

    folder_dict = {}  # 初始化
    thresh_hold = 30  # 假设10s

    while True:
        dic = get_url()
        textcontents = [x.strip().split(',')[0] for x in
                        open(pid_info_path, 'r').readlines()]  # 读取文本里的文件夹，假设格式 : 文件夹：pid号
        manage_dict(thresh_hold, pid)

        for dirpath, dirnames, filenames in os.walk(result_paths):
            for dir in dirnames:
                exist_img = (len(os.listdir(result_paths + '/' + dir)) >= 1)
                if not exist_img:
                    continue
                else:
                    if dir in folder_dict.keys():  # 如果在字典里
                        folder_dict[dir] = time.time()  # 更新时间
                    elif len(
                            folder_dict.keys()) < max_number and dir not in folder_dict.keys() and dir not in textcontents:  # 字典未满，并且文件夹不在字典和文本里
                        folder_dict[dir] = time.time()  # 添加文件夹并加入时间
                        with open(pid_info_path, 'a', encoding='utf-8') as f_w:
                            f_w.write(dir + ',' + str(pid) + '\n')
                            f_w.close()
                    elif len(folder_dict.keys()) == max_number:
                        break
                break
        if len(folder_dict.keys()) > 0:
            for name in list(folder_dict.keys()):
                path1 = result_paths + '/' + name
                # path1='result/001'
                # dirpath1='result/001'
                # dirnames1=<class 'list'>: [0]
                # filenames1=<class 'list'>:'001_102113.jpg','001_102114.jpg'...
                for dirpath1, dirnames1, filenames1 in os.walk(path1):
                    for filename in filenames1:  # filename='001_102113.jpg'
                        images_set.add(os.path.join(dirpath1, filename))
                        for im_path in list(images_set):  # images_set:'result/001\\001_102113.jpg'
                            # images_set = 列表[]
                            '''im_path = results/001\\001_182223.jpg'''
                            t1 = time.time()

                            img = cv2.imread(im_path, 1)
                            try:
                                img.shape
                                img  = retinaface.detect_image(img)
                                image, detection = np.array(list(yolo.detect_image(img)), dtype=object)

                                data = []
                                boxes = []
                                if len(detection):  # detection: <class 'list'>: [360, 0, 1031, 400, 'fire']
                                    for i in range(len(detection)):
                                        boxes.append(detection[i][:4])
                                        detections = dict()
                                        type = {"type": detection[i][-1]}   # 'fire'
                                        flag = {"flag": 3}  # 3
                                        pos = {"pos": ','.join(str(i) for i in (detection[i][:4]))}  # '360,0,1031,400'

                                        detections.update(type)
                                        detections.update(flag)
                                        detections.update(pos)
                                        data.append(detections)

                                    # new_time = time.strftime('%H%M%S', time.localtime())

                                    # img_path_img = im_path.split('/')[-1]
                                    #
                                    _img_path = im_path.split('/')[1]
                                    _img_path = _img_path.split('\\')[0]    #
                                    #
                                    # result_path_ = result_paths + '/' + name
                                    # newname = (result_path_+'/'+img_path_img)
                                    # if not os.path.exists(result_path_):
                                    #     os.makedirs(result_path_)
                                    #
                                    # size = glob.glob(result_path_+'/'+'*')

                                    result_path_ = result_paths + '/' + name

                                    if not os.path.exists(result_path_):
                                        os.makedirs(result_path_)

                                    size = glob.glob(result_path_ + '/' + '*')

                                    if len(size) < 1:
                                        post(image, data, dic, _img_path)
                                        images_set.remove(im_path)
                                        os.remove(im_path)

                                        t2 = time.time()
                                        print(t2 - t1)
                                    else:
                                        result = image_rmrept(image, result_path_)
                                        if result:
                                            # 图片经过检测，name修改,如 01_1.jpg ------ 01_1.jpg
                                            post(image, data, dic, _img_path)
                                            images_set.remove(im_path)
                                            os.remove(im_path)

                                            t3 = time.time()
                                            print(t3 - t1)
                                        else:
                                            images_set.remove(im_path)
                                            os.remove(im_path)
                                else:
                                    images_set.remove(im_path)
                                    os.remove(im_path)
                            except Exception as e:
                                print(e)
        else:
            print("jinchengjieshu")
            break
    os.system('kill -9 %s' % pid)


# os.kill(int(pid), signal.SIGINT)


def main():
    # 创建子进i程
    global path
    dir_set = set()
    read_account(pid_info_path)
    while True:

        new_time_1 = time.strftime('%H%M%S', time.localtime())

        if new_time_1 == str('235959'):
            read_account(rtmpurl_pid_path)

        #            for file in os.listdir(path):
        #               path_file = os.path.join(path, file)
        #               if os.path.isfile(path_file):
        #                    os.remove(path_file)

        with open(pid_info_path, 'r', encoding='utf-8') as f_r:
            dirs = f_r.readlines()
            if len(dirs) > 0:
                for dir in dirs:
                    dir_set.add(dir)
            else:
                dir_set.clear()
            f_r.close()
        if len(dir_set) != 0:
            if len(dir_set) % 5 == 0:
                ctx = torch.multiprocessing.get_context('spawn')
                # ctx = multiprocessing.get_context('spawn')
                print("开始检测1")
                son_p1 = ctx.Process(target=to_work, args=(), daemon=False)
                son_p1.start()
                time.sleep(10)

        else:
            flag = False
            for dirpath3, dirnames3, filenames3 in os.walk(result_paths):
                if len(filenames3) > 0:
                    flag = True
                    break
            if flag:

                ctx = torch.multiprocessing.get_context('spawn')
                print("开始检测2")
                son_p1 = ctx.Process(target=to_work, args=(), daemon=False)
                son_p1.start()
                time.sleep(60)
            else:
                continue


if __name__ == '__main__':
    """
    queue
    """
    _img_path = '001'
    main()
    # dic = get_url()
    # values = (dic.get(_img_path))
    # value = values.split(',')
    # print(value[0])
    # print(value[1])
    # print(type(value[1]))
    # print(value[2])
    #
    # print(value[3])
