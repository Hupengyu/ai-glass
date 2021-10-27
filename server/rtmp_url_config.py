from time import strftime
from datetime import datetime
import configparser
import os


MAX_INPUT_CAMERAS = 10
MAX_WAIT_TIME = 120 # SECONDSE
MAX_INVLAID_CAP_NUM = 5 # 最大无效图片获取次数
MAX_INVLAID_FRAME_NUM = 200 # 最大无效图片获取次数
rtpmurl_save_path = '../model_data/BoxStatu.txt'  #

rtpmurl_info_path = '../model_data/Project_pid.txt'  # used to send rtpm info to detection server

CAMERA_STATUS_SHUTDOWN = "statu00" #camera offline
CAMERA_STATUS_START = "statu01" #camera offline
class Rtpm():
    def __init__(self, rtpm_url, applydate, camera_id, backurl):
        self.rtpm_url = rtpm_url
        self.applydate = applydate
        self.camera_id = camera_id
        self.backurl = backurl
        #print(self.rtpm_url)
    def to_str(self):
        #print(type(self.applydate))
        return  self.rtpm_url + '\t' + self.applydate + '\t' + self.camera_id + '\t' + self.backurl


def gen_unique_cameraid():
    time_str = datetime.now().strftime('%Y%m%d%H%M%S%f')
    return time_str

def load_ai_glasses_rtpm_conf():
    rtpm_dict = dict()
    lines = open(rtpmurl_info_path, 'r').readlines()
   #print(lines)
    for line in lines:
        line = line.strip().strip('\r').strip('\n')
        #print(line)
        if line == '':
            continue
        strs = line.split('\t')
       #print(strs)
        #print(len(strs))
        if len(strs) > 2:
            rtpm = Rtpm(strs[0], strs[1], strs[2],strs[3])
            #print(type(rtpm))
            rtpm_dict[strs[0]] = rtpm
            #print(rtpm_dict[strs[0]]).
    print(rtpm_dict.items())
    return rtpm_dict

def save_rtpm_conf(rtpm_dict):
    f = open(rtpmurl_save_path, 'w')
    for (key, val) in rtpm_dict.items():
        f.write(val.to_str() + '\n')
    f.close()

def send_rtpm_info(rtpmurl, applydate, camera_status,backurl):
    with open(rtpmurl_info_path, 'w') as f:
        f.write(rtpmurl + ',' + applydate + ',' + camera_status+','+backurl)
        f.close()
    #size = os.path.getsize(rtpmurl_info_path)
    # if size != 0:
    #     n = 0
    #     with open(rtpmurl_info_path, 'r', encoding='utf-8') as f_r:
    #         texts = f_r.readlines()
    #         print(texts)
    #
    #
    #         for text in texts:
    #             if rtpmurl+','  in text:
    #                 print(rtpmurl+','  in text)
    #                 break
    #             if rtpmurl + ','  not in text:
    #                 n = n+1
    #         if n ==len(texts):
    #
    #             with open(rtpmurl_info_path, 'a+', encoding='utf-8') as f_x:
    #                 f_x.write(rtpmurl + ',' + applydate + ',' + camera_status + ',' + backurl )
    #                 f_x.close()
    #
    #
    #
    #             #f_w.close()
    #     #f_r.close()
    # else:
    #     with open(rtpmurl_info_path, 'w', encoding='utf-8') as f_w:
    #
    #         f_w.write(rtpmurl + ',' + applydate + ',' + camera_status + ',' + backurl + '\n')
    #         f_w.close()
def record_ai_glasses(rtpmurl, applydate, rtpm_dict , camera_status,backurl):
    #send_rtpm_info(rtpmurl, applydate, camera_status)

    if  rtpmurl in rtpm_dict:
        send_rtpm_info(rtpmurl, applydate, camera_status,backurl)
        return rtpm_dict, len(rtpm_dict), MAX_INPUT_CAMERAS, "000"
    else:
        if MAX_INPUT_CAMERAS == len(rtpm_dict):
            return rtpm_dict, len(rtpm_dict), MAX_INPUT_CAMERAS, "523"
        else:
            rtpm = Rtpm(rtpmurl, applydate, gen_unique_cameraid(),backurl)
            rtpm_dict[rtpmurl] = rtpm
            print(rtpm)
            '''
            f = open(rtpmurl_save_path, 'w')
            for (key, val) in rtpm_dict.items():
                f.write(val.to_str() + '\n')
            f.close()
            '''
            send_rtpm_info(rtpmurl, applydate, camera_status,backurl)
            #print(send_rtpm_info(rtpmurl, applydate, camera_status,backurl))

            return rtpm_dict, len(rtpm_dict), MAX_INPUT_CAMERAS, "000"
if __name__ == '__main__':

    rtpm_dict = load_ai_glasses_rtpm_conf()
    ai = record_ai_glasses(rtpmurl='rtmp://58.200.131.2:2020/livetv/hunantc', applydate='2020-03-20 20:26:31', rtpm_dict=rtpm_dict , camera_status='01',backurl="http://139.9.5.146:12072/webapi/api/aiReceiveRecord")
    print(rtpm_dict)
    # send_rtpm_info(rtpmurl='rtmp://49.233.13.40:1935/live/GL_5a4dda0c8d6741d4b1ef78543749d117_7FFD4CDd',
    #                applydate='2021/3/24 15:45:52',camera_status='statu01',backurl='http://49.233.13.40:12077/WebAPI/api/aiReceiveData')
    # print(send_rtpm_info)
