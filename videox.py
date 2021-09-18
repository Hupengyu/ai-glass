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
pid_info_path = './server/pid.txt'
rtmpurl_pid_path = './server/url_pids.txt'
path = 'imgg'
test_path = './test'
result_path = 'result'
result_paths = 'results'
yolo = YOLO()
alive = Value('b', False)
import os

def frame2base64(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame) #将每一帧转为Image
    #cv2.imwrite(image_path + '/'  + "1.jpg", frame)
    output_buffer = BytesIO() #创建一个BytesIO
    img.save(output_buffer, format='JPEG') #写入output_buffer
    byte_data = output_buffer.getvalue() #在内存中读取
    Base64 = base64.b64encode(byte_data) #转为BASE64
    return Base64
#去重
def image_rmrept(frame,pic_path):
    #reba = face_rmrept(frame, danger_list, face_path, pic_path)

    res = hash_rmrept(frame, pic_path)
    if res == 'same_pic':

        return False
    elif res == 'diff_pic':
        return True

def get_url():
    global backurl
    with open(rtmpurl_pid_path, 'r') as f:
        dic = {}
        lines = f.readlines()
        for line in lines:
            if line is None:
                break
            else:
                re = line.strip('\n').split(',')
                # for line in lines:
                key = re[1]
                value = re[0]
                backurl = re[2]
                dic.setdefault(key, value + ',' + backurl)
        f.close()
    return dic

def post(fras,data,url,backurl):

    headers = {'Content-Type':'application/json;charset=UTF-8'}
    value = {"base64": frame2base64(fras).decode('utf-8'),
             "checkdate":str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),
             "rtmpurl": url,
             "data":data
    }
    t2 = time.time()
    #param = json.dumps(value,cls=MyEncoder,indent=4)
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
# def manage_dict(thresh_hold):  # 处理字典超时方法
#    global yolo,folder_dict
#    while True:
#        delete = []
#        current_time = time.time()
#        for key, val in list(folder_dict.items()):
#            if (current_time - folder_dict[key]) >= thresh_hold:
#                delete.append(key)  # 删除超时文件夹
#        for i in range(len(delete)):
#            folder_dict.pop(delete[i])
#            lines = [l for l in open(pid_info_path, "r") if l.find(str(delete[i]) + ',') != 0]
#
#            with open(pid_info_path, 'w', encoding='utf-8') as f:
#                f.writelines(lines)
#                f.close()
#        if len(folder_dict) <= 0:
#            time.sleep(3)
def manage_dict(thresh_hold, pid):  # 处理字典超时方法
    global yolo,folder_dict
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
    global yolo,folder_dict
    dir_set = set()
    images_set = set()
    max_number = 5
    pid = os.getpid()

    folder_dict = {}  # 初始化
    thresh_hold = 15  # 假设10s


# def manage_dict(thresh_hold):  # 处理字典超时方法
    #

   #  manage_dict(thresh_hold)
   #  t1 = threading.Thread(target=manage_dict, args = (thresh_hold,),)  # 线程方法实时监督字典
   #  t1.start()


    while True:
        dic = get_url()

        textcontents = [x.strip().split(',')[0] for x in open(pid_info_path,'r').readlines()] # 读取文本里的文件夹，假设格式 : 文件夹：pid号
        manage_dict(thresh_hold, pid)

        for dirpath, dirnames, filenames in os.walk(path):
            for dir in dirnames:
                exist_img = (len(os.listdir(path + '/' + dir)) >= 1)
                if not exist_img:
                    continue
                else:
                    if dir in folder_dict.keys():  # 如果在字典里
                        folder_dict[dir] = time.time()  # 更新时间
                    elif len(
                            folder_dict.keys()) < max_number and dir not in folder_dict.keys() and dir not in textcontents:  # 字典未满，并且文件夹不在字典和文本里
                        folder_dict[dir] = time.time() # 添加文件夹并加入时间
                        with open(pid_info_path, 'a', encoding='utf-8') as f_w:
                            f_w.write(dir + ',' + str(pid) + '\n')
                            f_w.close()
                    elif len(folder_dict.keys()) == max_number:
                        break
                break
        if len(folder_dict.keys()) >0:
            for name in list(folder_dict.keys()):
                path1 = path + '/' + name
                for dirpath1, dirnames1, filenames1 in os.walk(path1):
                    for filename in filenames1:
                        images_set.add(os.path.join(dirpath1, filename))
                        for im_path in list(images_set):  # "images_set:"
                            # images_set = 列表[]
                            t1 = time.time()

                            img = cv2.imread(im_path, 1)



                            try:
                                img.shape
                                image, detection = np.array(list(yolo.detect_image(img)), dtype=object)
                            except:
                                continue

                            data = []
                            boxes = []
                            if len(detection):
                                for i in range(len(detection)):
                                    boxes.append(detection[i][:4])
                                    detections = dict()
                                    type = {"type": detection[i][-1]}
                                    flag = {"flag": 3}
                                    pos = {"pos": ','.join(str(i) for i in (detection[i][:4]))}

                                    detections.update(type)
                                    detections.update(flag)
                                    detections.update(pos)
                                    data.append(detections)

                                new_time = time.strftime('%H%M%S', time.localtime())

                                # if not os.path.exists(result_paths):
                                #     os.makedirs(result_paths)
                                #cv2.imwrite(result_paths + '/' + str(pid) + "_" + new_time + ".jpg", image)
                                img_path_img = im_path.split('/')[-1]

                                _img_path = im_path.split('/')[1]

                                urls = (dic.get(_img_path))
                                print(urls)
                                rtmpurl = urls.split(',')[0]
                                backurl = urls.split(',')[1]

                                result_path_ = result_path + '/' + name
                                newname = (result_path_+'/'+img_path_img)
                                if not os.path.exists(result_path_):
                                    os.makedirs(result_path_)

                                size = glob.glob(result_path_+'/'+'*')

                                # img_path = im_path.split('\\')[0]
                                # img_path_img = im_path.split('\\')[-1]
                                #
                                # _img_path = img_path.split('/')[1]
                                #
                                # url = (dic.get(_img_path))
                                #
                                # result_path_ = result_path + '/' + name
                                # newname = (result_path_ + '/' + img_path_img)
                                # if not os.path.exists(result_path_):
                                #     os.makedirs(result_path_)
                                #
                                # size = glob.glob(result_path_ + '/' + '*')
                                if len(size) <1:
                                    post(image, data, rtmpurl, backurl)
                                    images_set.remove(im_path)
                                    #os.rename(im_path, newname)
                                    shutil.move(im_path, newname)
                                    t2 = time.time()
                                    print(t2 - t1)
                                else:
                                    result = image_rmrept(image, result_path_)
                                    if result:
                                        # 图片经过检测，name修改,如 01_1.jpg ------ 01_1.jpg
                                        post(image, data, rtmpurl, backurl)
                                        images_set.remove(im_path)

                                        shutil.move(im_path, newname)
                                        t3 = time.time()
                                        print(t3 - t1)
                                    else:
                                        images_set.remove(im_path)
                                        os.remove(im_path)
                            else:
                                images_set.remove(im_path)
                                os.remove(im_path)


     
        else:

            print("jinchengjieshu")
            break
    os.system('kill -9 %s' % pid)

            #os.kill(int(pid), signal.SIGINT)
def del_file(filepath):
    print("hello")
    listdir = os.listdir(filepath)  # 获取文件和子文件夹
    print(listdir)
    for dirname in listdir:
        dirname = filepath + "//" + dirname
        if os.path.isfile(dirname):  # 是文件
            print(dirname)
            shutil.rmtree(dirname)  # 删除文件
        elif os.path.isdir(dirname):  # 是子文件夹
            print(dirname)
            dellist = os.listdir(dirname)
            for f in dellist:  # 遍历该子文件夹
                file_path = os.path.join(dirname, f)
                if os.path.isfile(file_path):  # 删除子文件夹下文件
                    os.remove(file_path)
                elif os.path.isdir(file_path):  # 强制删除子文件夹下的子文件夹
                    shutil.rmtree(file_path)
            

def main():
    # 创建子进i程
    global path
    dir_set = set()
    read_account(pid_info_path)
    while True:

        new_time_1 = time.strftime('%H%M%S', time.localtime())

        if new_time_1 == str('235959'):
            del_file(result_path)


#            for file in os.listdir(path):
#               path_file = os.path.join(path, file)
#               if os.path.isfile(path_file):
#                    os.remove(path_file)


        with open(pid_info_path, 'r', encoding='utf-8') as f_r:
            dirs = f_r.readlines()
            if len(dirs) >0:
                for dir in dirs:
                    dir_set.add(dir)
            else:
                dir_set.clear()
            f_r.close()
        if len(dir_set) !=0:
            if len(dir_set) % 5 == 0:
                ctx = torch.multiprocessing.get_context('spawn')
                # ctx = multiprocessing.get_context('spawn')
                print("开始检测1")
                son_p1 = ctx.Process(target=to_work, args=(),daemon=False)
                son_p1.start()
                time.sleep(10)

        else:
            flag = False
            for dirpath3, dirnames3, filenames3 in os.walk(path):
                if len(filenames3) > 0:
                    flag = True
                    break
            if flag:


                    ctx = torch.multiprocessing.get_context('spawn')
                    print("开始检测2")
                    son_p1 = ctx.Process(target=to_work, args=(),daemon=False)
                    son_p1.start()
                    time.sleep(60)
            else:
                continue




if __name__ == '__main__':
    """
    queue
    """
    main()
