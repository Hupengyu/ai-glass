import requests
import base64
import cv2


def base_img(img_im):
    return base64.b64encode(cv2.imencode('.jpg',img_im)[1]).decode()


pic_path = r'5005008401.jpg'
img_im = cv2.imread(pic_path)
#print(img_im)

#print(base64_img)
#status =[{'002': 'K:/JKY-AI-CX/1.mp4'},{'001':'K:/JKY-AI-CX/MAH00060.MP4'}]
# status =[{'001': 'rtsp://admin:1234567@192.168.1.100:554/Streaming/Channels/101?transportmode=unicast'},
#          {'002':'rtsp://admin:admin1234567@192.168.1.102:454/Streaming/Channels/101?transportmode=unicast'}]
#status =[{'001':'rtsp://admin:jky123456@192.168.2.188'}]
status =[{'001':'rtsp://admin:jky123456@192.168.2.188'}]
#status =[{'001':'rtsp://admin:jky123456@192.168.2.188'},{'002':'rtsp://admin:jky123456@192.168.0.33'},{'003':'rtsp://admin:jky123456@192.168.0.22'}]



# data = {"rtpmurl":"rtmp://49.233.13.40:1935/live/GL_a12871d2baaa2f74cb04b462a4f6ef502eaddadasdsadwad",
#         "applydate":"2020-03-20 20:26:31","camera_status":"statu01",
#         "backurl":"http://49.233.13.40:12077/WebAPI/api/aiReceiveData"}

data2 = {'ProjectID': "a79ec4e6-472d-413e-9585-66121b731cba",
        'BoxCode':'001',
        'backurl':"http://49.233.13.40:12077/WebAPI/api/aiReceiveData",
        'status':status}


#
#
#url = "http://119.3.185.115:5903/server_v4"
url = "http://localhost:8777/server_v4"

headers = {'Content-Type':'application/json;charset=UTF-8'}

res = requests.request("post", url, json=data2, headers=headers)
print(res.status_code)
print(res.text)


"""
图片压缩方法更改
视频流地址查询缘由
"""