from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, jsonify, request
import os
import cv2
import time
import configparser
import requests
import multiprocessing as mp
import os
#from lshw import parseDmi
from lshw import backurl,parseDmi,get_ip

BoxStatu = '../model_data/BoxStatu.txt'
app = Flask('jky_c_server')
basedir = os.path.abspath('.')
url_bianhao = 0
url_paths = '../model_data/url_cameras.txt'
result_path = '../results'
camera_ip_set = {}
ip_id = 0
BoxID = parseDmi()
#BoxID = '14221254'
backurl,camera_url = backurl()
ip = get_ip()
code_now = 0
result_now = dict()



def post(BoxID,backurl,ip):
    #data = 'http://192.168.2.181:8777/AiBox'
    data = 'http://'+ip + ':8777/AiBox'
    headers = {'Content-Type':'application/json;charset=UTF-8'}
    value = {"boxID": BoxID,
             "URL": data

             }
    t2 = time.time()
    #print(value)
    res = requests.post(url=backurl,json=value,headers=headers)
    print(res.text)
    result = res.json()
    return result
def post_statu():
    # while True:
    global backurl,camera_url,BoxID,ip,code_now,result_now
    n = 0
    while True:

        if code_now == 1:
            now_time = time.strftime('%H%M%S', time.localtime())
            if now_time == str('235959'):
                result = post(BoxID, camera_url, ip)  
                code = int(result.get('code'))
                data = result.get('data')
                project_id = (data.get('Box')).get('PRJId')
                cameras = (data.get('Box')).get('Camera')
                CamaraAddresses = []
                for i in cameras:
                    CamaraAddress = i.get('CamaraAddress')
                    CamaraAddresses.append(CamaraAddress + '\n')
                with open(BoxStatu, 'w', encoding='utf-8') as fix:
                    fixid = project_id + ',' + BoxID + ',' + backurl
                    fix.writelines(fixid)
                    fix.close()

                with open(url_paths, 'w', encoding='utf-8') as p:
                    p.writelines(CamaraAddresses)
                    p.close()
                time.sleep(5)
                code_now = code
                result_now = result

        else:
            time.sleep(5)
            result = post(BoxID, camera_url, ip)  
            code = int(result.get('code'))
            code_now = code
            result_now = result
def open_statu():
    global backurl, camera_url, BoxID, ip,code_now,result_now
    # BoxID = '14221254'
    result = post(BoxID, camera_url, ip) 
    code = int(result.get('code'))  # 0 1 -1
    if code == 1:
        data = result.get('data')
        print(data)
        project_id = (data.get('Box')).get('PRJId')
        print(project_id)
        cameras = data.get('Camera')
        print(cameras)
        CamaraAddresses = []
        for i in cameras:
            CamaraAddress = i.get('CamaraAddress')
            CamaraAddresses.append(CamaraAddress + '\n')
        with open(BoxStatu, 'w', encoding='utf-8') as fix:
            fixid = project_id + ',' + BoxID + ',' + backurl
            fix.writelines(fixid)
            fix.close()

        with open(url_paths, 'w', encoding='utf-8') as p:
            p.writelines(CamaraAddresses)
            p.close()
    time.sleep(5)

    code_now = code
    result_now = result

if __name__ == '__main__':
    open_statu()
    #post_statu()
