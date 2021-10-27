#!/usr/bin/env python
# encoding: utf-8

'''
收集主机的信息：
主机名称、IP、系统版本、服务器厂商、型号、序列号、CPU信息、内存信息
'''
#import fcntl
import socket, struct
import sys
import configparser
from subprocess import Popen, PIPE
import requests
import json
def backurl():
    config = configparser.ConfigParser()
    config.read(r'../model_data/conf.ini', encoding='UTF-8')
    backurl = config.get("address",'backurl')
    camera_url = config.get("address",'camera_url')
    return backurl,camera_url

def parseDmi():
    p = Popen(['lshw'], stdout=PIPE)
    print(p)
    data = p.stdout.read()
    data = str(data, encoding="utf-8")

    parsed_data = []
    new_line = ''
    data = [i for i in data.split('\n') if i]
    for line in data:
        if line[0].strip():
            parsed_data.append(new_line)
            new_line = line + '\n'
        else:
            new_line += line + '\n'
    parsed_data.append(new_line)
    parsed_data = [i for i in parsed_data if i]
    parsed_data = [i for i in parsed_data[0].strip('').split('\n') if 'serial' in i]
    #dmi_dic = dict([i.strip().split(':') for i in parsed_data])
    # for serial in parsed_data:
    serial,boxid = parsed_data[0].replace(' ','').split(':')


        #print(len(serial))
        #print(type(serial))

        # se,num_id = serial.split(':')jide
        # print(len(num_id))
        # print(num_id[0])
        # num_id.strip('·')
        # if num_id.isnumeric():
        #     dic['serial'] = dmi_dic['Serial Number'].strip()

    return boxid

def ipconfig():

    p = Popen(['./ngrok tcp 8777'], stdout=PIPE)
    data = p.stdout.read()
    data = str(data, encoding="utf-8")
    print(data)
    parsed_data = []
    new_line = ''
    data = [i for i in data.split('\n') if i]
    for line in data:
        if line[0].strip():
            parsed_data.append(new_line)
            new_line = line + '\n'
        else:
            new_line += line + '\n'
    parsed_data.append(new_line)

    # dic = {}
    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # ipaddr = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', ifname[:15]))[20:24])
    # dic['ip_out'] = ipaddr
    return new_line

# import re
# import time
def get_ip():
    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s.connect(('8.8.8.8', 88))
    # ip = s.getsockname()[0]
    ip = json.loads(requests.get('https://api.ipify.org/?format=json').text)['ip']

    return ip
if __name__ == '__main__':
    # ip = get_ip()
    # print(ip)
    ip = get_ip()
    print(ip)