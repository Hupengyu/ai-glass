import requests
import base64
#import cv2


# def base_img(img_im):
#     return base64.b64encode(cv2.imencode('.jpg',img_im)[1]).decode()
#
#
# pic_path = r'5005008401.jpg'
# img_im = cv2.imread(pic_path)
# #print(img_im)

#print(base64_img)
#status =[{'002': 'K:/JKY-AI-CX/1.mp4'},{'001':'K:/JKY-AI-CX/MAH00060.MP4'}]
# status =[{'001': 'rtsp://admin:1234567@192.168.1.100:554/Streaming/Channels/101?transportmode=unicast'},
#          {'002':'rtsp://admin:admin1234567@192.168.1.102:454/Streaming/Channels/101?transportmode=unicast'}]
#status =[{'001':'rtsp://admin:jky123456@192.168.2.188'}]
#status =[{'001':'rtsp://admin:jky123456@192.168.2.188'},{'002':'rtsp://admin:jky123456@192.168.0.22'}]
status =[{'001':'rtsp://admin:jky123456@192.168.2.188'},{'002':'rtsp://admin:jky123456@192.168.0.33'},{'003':'rtsp://admin:jky123456@192.168.0.22'}]



# data = {"rtpmurl":"rtmp://49.233.13.40:1935/live/GL_a12871d2baaa2f74cb04b462a4f6ef502eaddadasdsadwad",
#         "applydate":"2020-03-20 20:26:31","camera_status":"statu01",
#         "backurl":"http://49.233.13.40:12077/WebAPI/api/aiReceiveData"}

#data ={"code":"1","msg":None,"data":{"Box":{"ID":"915CA2D3-DDA3-40BF-9BE0-240EC2621A48","BoxName":"","BoxID":"1421021043768","PRJId":"c44d7439-619e-4eb4-81bd-62401020cd58","CJR":"bc75bd6d-994e-48f3-8c8d-090aae23c53f","CJSJ":"2021-10-08T09:22:29","XGR":"","XGSJ":"2021-10-08T09:22:29","IsValid":True},"Camera":[{"ID":"96bcd688-4b27-43be-9fa3-dd09dfccf7af","BoxId":"915CA2D3-DDA3-40BF-9BE0-240EC2621A48","CamaraName":"liujialin","CamaraAddress":"rtsp://admin:jky123456@192.168.0.33","Status":0,"CamaraNO":1,"CJR":"","CJSJ":"2021-10-08T09:22:46","XGR":"","XGSJ":None,"IsValid":True}]}}
data = {"code":1,"msg":None,"data":{"Box":{"ID":"a79ec4e6-472d-413e-9585-66121b731cba","BoxName":"","BoxID":"1421021043768","PRJId":"a79ec4e6-472d-413e-9585-66121b731cba","CJR":"bc75bd6d-994e-48f3-8c8d-090aae23c53f","CJSJ":"2021-10-08T09:22:29","XGR":"","XGSJ":"2021-10-08T09:22:29","IsValid":True},"Camera":[{"ID":"5d964983-21ba-4fea-9980-bb294694c60f","BoxId":"915CA2D3-DDA3-40BF-9BE0-240EC2621A48","CamaraName":"222222222","CamaraAddress":"rtsp://admin:jky123456@192.168.0.33","Status":0,"CamaraNO":3,"CJR":"","CJSJ":"2021-10-18T14:36:22","XGR":"","XGSJ":None,"IsValid":True},{"ID":"8317636c-c2f3-4180-99f6-b98842ea18c2","BoxId":"915CA2D3-DDA3-40BF-9BE0-240EC2621A48","CamaraName":"33333333","CamaraAddress":"rtsp://admin:jky123456@192.168.2.188","Status":0,"CamaraNO":4,"CJR":"","CJSJ":"2021-10-18T14:36:33","XGR":"","XGSJ":None,"IsValid":True},{"ID":"a1f6617f-f5a7-45e1-9d42-c5a7daf6f473","BoxId":"915CA2D3-DDA3-40BF-9BE0-240EC2621A48","CamaraName":"1111","CamaraAddress":"rtsp://admin:jky123456@192.168.0.22","Status":0,"CamaraNO":2,"CJR":"","CJSJ":"2021-10-18T14:36:10","XGR":"","XGSJ":None,"IsValid":True}]}}


#
#
url = "http://192.168.2.174:80/AiBox"
#url = "http://125.34.19.76:80/AiBox"
#url = 'http://125.34.17.139:8080/AiBox'
#url = '2.tcp.ngrok.io:10549'

headers = {'Content-Type':'application/json;charset=UTF-8'}

res = requests.request("post",url,json=data, headers=headers)
print(res.status_code)
print(res.text)


"""
图片压缩方法更改
视频流地址查询缘由
"""