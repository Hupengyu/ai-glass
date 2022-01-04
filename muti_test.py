import subprocess as sp
import time
from yolo3_rtsp import YOLO
import cv2 as cv
from threading import Thread
from server.lshw import urls, parseDmi
import multiprocessing
import queue

rtsp_init = []
confi = False
rtsp_path = 'model_data/url_cameras.txt'
fixid = 'model_data/BoxStatu.txt'
yolo = YOLO()
Boxid = parseDmi()
backurl, posturl, rtspurl = urls()
# ffmpeg command
frame_queue = queue.Queue()


def push_flush_1(camera_path, frame_queue):
    camera_path = camera_path
    frame_queue = frame_queue
    rtmpUrl = rtspurl + Boxid + camera_path
    cap = cv.VideoCapture(camera_path)
    "rtmp://119.3.185.115:8777/stream/rtsp://admin:jky123456@192.168.2.188"
    # Get video information
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    command = ['ffmpeg',
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
    print('进入class')
    start_thread()

    def video(self):
        # = 'rtsp://admin:jky123456@192.168.2.188'

        # 管道配置

        # read webcamera
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Opening camera is failed")
                break
            frame_queue.put(frame)

    def push_frame(self):
        while True:
            if len(command) > 0:
                # 管道配置，其中用到管道
                p = sp.Popen(command, stdin=sp.PIPE)
                break
        while True:
            if not frame_queue.empty():
                # 从队列中取出图片
                frame = frame_queue.get()
                frame = yolo.detect_video(frame)
                p.stdin.write(frame.tostring())

    def start_thread(self):
        print('进入start')
        thread1 = Thread(target=video())
        thread1.start()
        thread2 = Thread(target=push_frame())
        thread2.start()





def run():
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
                    print(URL)
                    send, get = multiprocessing.Pipe()
                    process_1 = multiprocessing.Process(target=push_flush_1, args=(URL, frame_queue))
                    process_2 = multiprocessing.Process(target=push_flush_2, args=(URL, frame_queue))
                    process_1.start()
                    process_2.start()
                    process_1.join()
                    process_2.join()


if __name__ == '__main__':
    run()
    # main()
