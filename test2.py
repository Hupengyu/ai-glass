import cv2
import base64
import time
import multiprocessing as mp
import os
from PIL import Image

path = 'server/ProjectID.txt'
result_path = 'results'

def image_put(q, user, pwd, ip, channel=1):
    print(ip)
    webcm = ip.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
    if webcm:

        ret,cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel))

    else:
        cap = cv2.VideoCapture(ip)
        print("???????????????")

    while True:

        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)

camera_ip_set = {}
ip_id = 0
def image_get(q, camera_ip,camera_ip_set):
    print(camera_ip+'?????')

    c = 1
    while True:
        #print(camera_ip_set)
        #cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
        new_time = time.strftime('%H%M%S', time.localtime())
        name = camera_ip_set[camera_ip] + '_' + new_time + '.jpg'
        path_a = os.path.join(result_path, camera_ip_set[camera_ip], name)

        frame = q.get()
        c = c +1
        timeF = 60
        if frame is not None:
            # c = 1
            # timeF = 10

            if (c % timeF == 0):
                cv2.imwrite(path_a,frame)

        else:
            break

# def run_single_camera():
#     user_name, user_pwd, camera_ip = "admin", "admin123456", "192.168.35.121"
#
#     mp.set_start_method(method='spawn')  # init
#     queue = mp.Queue(maxsize=2)
#     processes = [mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)),
#                  mp.Process(target=image_get, args=(queue, camera_ip))]
#
#     [process.start() for process in processes]
#     [process.join() for process in processes]



def run_multi_camera(paths):
    global ip_id,camera_ip_set

    # user_name, user_pwd = "admin", "password"
    user_name, user_pwd = "admin", "admin123456"
    camera_ip_l = []
    with open(paths, 'r', encoding='utf-8') as txt:
        path_x = txt.readlines()
    for path in path_x:
        pa = '\n'.join(path.split())
        camera_ip_l.append(pa)


    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=10) for _ in camera_ip_l]
    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        if camera_ip not in camera_ip_set.keys():
            ip_id += 1
            camera_ip_set.setdefault(camera_ip, "%03d" % ip_id)
            print(camera_ip_set)
            if not os.path.exists(os.path.join(result_path, camera_ip_set[camera_ip])):
                os.makedirs(os.path.join(result_path, camera_ip_set[camera_ip]), exist_ok=True)
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=image_get, args=(queue, camera_ip,camera_ip_set)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    # run_single_camera()
    run_multi_camera(path)

