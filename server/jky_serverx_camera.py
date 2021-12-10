from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, jsonify, request
import time
import configparser
import requests
import os
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
ip = '125.34.17.139'
code_now = 0
result_now = dict()

def read_account(filename):
    with open(filename, 'r+', encoding='utf-8') as f:
        res = f.readlines()
        f.seek(0)
        f.truncate()
        f.close()


@app.route('/AiBox', methods=['POST', 'GET'])
def api_upload():
    global url_bianhao,url_bianhao
    if request.method == 'GET':
        print("hello world")
        return "success"
    else:
        print('hello world')
        # try:

        datas = request.json['data']
        code = request.json['code']
        print(code)

   #      datas = {'code': 1, 'msg': None, 'data': {
   #          'Box': {'ID': 'BEF2A6AC-E9BC-48C1-9519-E6FADFCB5C3F', 'BoxName': '', 'BoxID': 'AAA',
   #                  'PRJId': 'd26418c9-1c5c-46e0-8f53-4190ce1363e0', 'CJR': 'bc75bd6d-994e-48f3-8c8d-090aae23c53f',
   #                  'CJSJ': '2021-09-27T11:29:00', 'XGR': '', 'XGSJ': '2021-09-27T11:29:00', 'IsValid': True},
   #          'Camera': [{'ID': '11920b5a-d56a-4bb8-8ee9-d246be46a03b', 'BoxId': 'BEF2A6AC-E9BC-48C1-9519-E6FADFCB5C3F',
   #                      'CamaraName': 'BBB', 'CamaraAddress': 'BBBBBB', 'Status': 0, 'CamaraNO': 2, 'CJR': '',
   #                      'CJSJ': '2021-09-27T11:50:14', 'XGR': '', 'XGSJ': None, 'IsValid': True},
   #                     {'ID': '5a658d30-964d-4d48-8cbd-8ae46297bfae', 'BoxId': 'BEF2A6AC-E9BC-48C1-9519-E6FADFCB5C3F',
   #                      'CamaraName': 'AAA', 'CamaraAddress': 'DDD', 'Status': 0, 'CamaraNO': 1, 'CJR': '',
   #                      'CJSJ': '2021-09-27T11:34:33', 'XGR': '', 'XGSJ': None, 'IsValid': True}]}}
   #
   #      """
   # data = {'code': 1, 'msg': None, 'data': {'Box': {'ID': 'BEF2A6AC-E9BC-48C1-9519-E6FADFCB5C3F', 'BoxName': '', 'BoxID': 'AAA', 'PRJId': 'd26418c9-1c5c-46e0-8f53-4190ce1363e0', 'CJR': 'bc75bd6d-994e-48f3-8c8d-090aae23c53f', 'CJSJ': '2021-09-27T11:29:00', 'XGR': '', 'XGSJ': '2021-09-27T11:29:00', 'IsValid': True}, 'Camera': [{'ID': '11920b5a-d56a-4bb8-8ee9-d246be46a03b', 'BoxId': 'BEF2A6AC-E9BC-48C1-9519-E6FADFCB5C3F', 'CamaraName': 'BBB', 'CamaraAddress': 'BBBBBB', 'Status': 0, 'CamaraNO': 2, 'CJR': '', 'CJSJ': '2021-09-27T11:50:14', 'XGR': '', 'XGSJ': None, 'IsValid': True}, {'ID': '5a658d30-964d-4d48-8cbd-8ae46297bfae', 'BoxId': 'BEF2A6AC-E9BC-48C1-9519-E6FADFCB5C3F', 'CamaraName': 'AAA', 'CamaraAddress': 'DDD', 'Status': 0, 'CamaraNO': 1, 'CJR': '', 'CJSJ': '2021-09-27T11:34:33', 'XGR': '', 'XGSJ': None, 'IsValid': True}]}}
   #
   #      """
        print(datas)
        config = configparser.ConfigParser()
        config.read(r'../model_data/conf.ini', encoding='UTF-8')

            #data.get
        if code ==1:
            #data =datas.get('data')
            Box = datas.get('Box')
            cameras = datas.get('Camera')
            ProjectID = Box.get('ID')
            BoxID =  Box.get('BoxID')
            CamaraAddresses = []
            for i in cameras:
                CamaraAddress = i.get('CamaraAddress')
                CamaraAddresses.append(CamaraAddress+'\n')
       # print(Camera)
            with open(BoxStatu, 'w', encoding='utf-8') as fix:
                fixid = ProjectID +','+ BoxID +','+ backurl
                fix.writelines(fixid)
                fix.close()

            with open(url_paths,'w', encoding='utf-8') as p:
                p.writelines(CamaraAddresses)
                p.close()




            return jsonify({"code": "1", "msg": "success"})

        else:
            return jsonify({"code": "-1","msg":"error"})


if __name__ == '__main__':
    read_account(BoxStatu)
    read_account(url_paths)
    #app.run(debug=True, host=ip, port=8777)
    app.run(debug=True, host='192.168.2.195', port=8777)
