import subprocess as sp
import time
from yolo3_rtsp import YOLO
import cv2 as cv
from threading import Thread
from server.lshw import urls, parseDmi
import multiprocessing
from multiprocessing import Process, Manager
import gc

rtsp_init = []
confi = False
rtsp_path = './model_data/url_cameras.txt'
fixid = './model_data/BoxStatu.txt'
yolo = YOLO()
Boxid = parseDmi()
backurl, posturl, rtspurl, photourl = urls()
# ffmpeg command
# frame_queue = queue.Queue()
from multiprocessing import Queue

q = Manager().list()


class push_flush_1(object):
    def __init__(self, camera_path, frame_queue, top):
        self.top = top
        self.camera_path = camera_path
        self.frame_queue = frame_queue
        rtmpUrl = rtspurl + Boxid + '/' + camera_path
        self.cap = cv.VideoCapture(camera_path)
        "rtmp://119.3.185.115:8777/stream/rtsp://admin:jky123456@192.168.2.188"
        # Get video information
        fps = int(self.cap.get(cv.CAP_PROP_FPS))
        width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.command = ['ffmpeg',
                        '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(width, height),
                        '-r', str(fps),
                        '-i', '-',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'ultrafast',
                        '-f', 'flv',
                        rtmpUrl]
        self.start_thread()

    def video(self):
        # = 'rtsp://admin:jky123456@192.168.2.188'

        # 管道配置

        # read webcamera
        while self.cap.isOpened():
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                print("Opening camera is failed")
                break
            if len(self.frame_queue) < 2:
                self.frame_queue.append(frame)
            else:
                time.sleep(0.05)

            t1 = time.time() - t0

    def push_frame(self):

        while True:
            if len(self.command) > 0:
                # 管道配置，其中用到管道：
                p = sp.Popen(self.command, stdin=sp.PIPE)
                break
        while True:
            t2 = time.time()
            if len(self.frame_queue) >= 2:
                frame = self.frame_queue.pop(-1)

                # 从队列中取出图片
                frame = yolo.detect_video(frame)
                t3 = time.time() - t2
                p.stdin.write(frame.tostring())

    def start_thread(self):
        thread1 = Thread(target=self.video, args=())
        thread1.start()
        thread2 = Thread(target=self.push_frame, args=())
        thread2.start()


def main():
    global rtsp_init

    while True:

        with open(rtsp_path, 'r', encoding='utf-8') as f_f:  # 打开盒子配置文件
            rtsp_now = f_f.readlines()

        if len(rtsp_now) > 0:
            rtsp_now = [x.strip('\n') for x in rtsp_now]
            if rtsp_init == rtsp_now:  # 如果推流txt 有内容 如果内容无变化
                continue
            else:
                rtsp_init = rtsp_now
                time.sleep(5)
                URLS = rtsp_init

                for URL in URLS:
                    URL = URL.split(',')[0]
                    ctx = multiprocessing.get_context('spawn')
                    thread = ctx.Process(target=push_flush_1, args=(URL, q, 2))
                    thread.start()


if __name__ == '__main__':
    main()
