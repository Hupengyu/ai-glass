import os
fixedid = 'Fixed_ID.txt'
rtmpurl_pid_path = 'c_camera.txt'
basedir = os.path.abspath('.')
url_bianhao = 0
config_camera = 'c_camera.txt'
import cv2
import time
import multiprocessing as mp
import os
import operator



path = 'url_pids.txt'
result_path = '../results'
result_paths = '../result'
camera_ip_set = {}
ip_id = 0
import glob
import shutil
def image_put(q, user, pwd, ip, channel=1):
    webcm = ip.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
    if webcm:
        ret,cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel))
        """
        rtsp://admin:1234567@192.168.1.100:554/Streaming/Channels/101?transportmode=unicast    
        """
    else:
        cap = cv2.VideoCapture(ip)
        print('视频读取图片')
    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)

def image_get(q, camera_ip,camera_ip_set,camera_ip_l):
    global path,path_all
    c = 1
    if not os.path.exists(os.path.join(result_paths, camera_ip_set[camera_ip])):
        os.makedirs(os.path.join(result_paths, camera_ip_set[camera_ip]))
    while True:

        new_time = time.strftime('%H%M%S', time.localtime())
        name = camera_ip_set[camera_ip] + '_' + new_time + '.jpg'
        path_a = os.path.join(result_path, camera_ip_set[camera_ip], name)
        endpath = os.path.join(result_paths, camera_ip_set[camera_ip], name)
        frame = q.get()
        timeF = 60
        if frame is not None:
            if (c % timeF == 0):
                cv2.imwrite(path_a,frame)
                shutil.move(path_a, endpath)
                print(path_a)
            c = c +1

        else:
            camera_ip_l.remove(camera_ip)
            break
def run_multi_camera():
    global ip_id,camera_ip_set
    path_all = []
    user_name, user_pwd = "admin", "jky123456"
    try:
        while True:
            size = os.path.getsize(path)
            # print(size)
            if size == 0:
                continue
            else:
                with open(path, 'r', encoding='utf-8') as f_r:
                    texts = f_r.readlines()
                    path_url = []
                    for pathaa in texts:
                        pathaa = '\n'.join(pathaa.split())
                        path_url.append(pathaa)
                    if operator.eq(path_url, path_all):
                        continue
                    else:
                        print("!!!!!!!")
                        path_all.clear()
                        for pathas in path_url:
                            print(pathas)
                            # pathaa = '\n'.join(pathaa.split())
                            path_all.append(pathas)
                        print(path_all)
    # user_name, user_pwd = "admin", "password"
                        if len(path_all)>0:

                            camera_ip_l = path_all
                            print(camera_ip_l)
                        # with open(paths, 'r', encoding='utf-8') as txt:
                        #     path_x = txt.readlines()
                        # for path in path_x:
                        #     pa = '\n'.join(path.split())
                        #     camera_ip_l.append(pa)


                            mp.set_start_method(method='spawn')  # init
                            queues = [mp.Queue(maxsize=10) for _ in camera_ip_l]
                            processes = []
                            for queue, camera_ip in zip(queues, camera_ip_l):
                                if camera_ip not in camera_ip_set.keys():
                                    ip_id += 1
                                    camera_ip_set.setdefault(camera_ip, "%03d" % ip_id)
                                    if not os.path.exists(os.path.join(result_path, camera_ip_set[camera_ip])):
                                        os.makedirs(os.path.join(result_path, camera_ip_set[camera_ip]), exist_ok=True)
                                processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
                                processes.append(mp.Process(target=image_get, args=(queue, camera_ip,camera_ip_set,camera_ip_l)))

                            for process in processes:
                                process.start()
                            for process in processes:
                                process.join()

    except Exception as e:
        print(e)
if __name__ == '__main__':
    run_multi_camera()
