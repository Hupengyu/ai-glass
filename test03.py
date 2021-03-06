import time

import cv2
import numpy as np
from PIL import Image
from retinaface import Retinaface
from yolo3_m import YOLO

yolo = YOLO()
#capture=cv2.VideoCapture('rtsp://admin:jky123456@192.168.0.22')
capture=cv2.VideoCapture('https://open.ys7.com/v3/openlive/F68456476_1_1.m3u8?expire=1665998854&id=371711193109610496&t=215c8bcf22a5e55b9343cd34fd316dc670c1355c2116e31223b89da68e9ceb47&ev=100')

# 调用电脑自带的摄像头这里是0，调用外接摄像头的话是1
fps = 0.0
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    print(type(frame))

    # 转变成Image
    # 进行检测
    #image, detection = np.array(list(yolo.detect_image(frame)), dtype=object)
    # RGBtoBGR满足opencv显示格式
    #print(detection)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # (0, 40)表示显示的文字坐标， cv2.FONT_HERSHEY_SIMPLEX是字体，(0, 255, 0)是字体颜色
    cv2.imshow("video",frame)

    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break