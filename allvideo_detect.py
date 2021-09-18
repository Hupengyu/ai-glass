import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from yolo3_x import YOLO
yolo = YOLO()
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,LoadWebcam
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression2
from utils.plots import plot_one_box
# from utils.plots_chinese import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import base64
import requests
from threading import Thread
import numpy as np
rtmp_url = []
back_url = []

def thread_read_backurl():
    global rtmp_url,back_url
    while True:
        temp_rtmp_url = []
        temp_back_url = []
        with open('./backurl.txt', 'r', encoding='utf-8') as txt:
            lines = txt.readlines()
            length = len(lines)
            for i in range(length):
                if i % 2 == 0:
                    temp_rtmp_url.append(lines[i].strip())
                    if i+1 > length - 1:
                        break
                    temp_back_url.append(lines[i + 1].strip())
        txt.close()
        rtmp_url = temp_rtmp_url
        back_url = temp_back_url
        time.sleep(3)

def frame2base64(img_im):
    return base64.b64encode(cv2.imencode('.jpg',img_im)[1]).decode()

def post(data,save_path):
    label_dict = {}
    for i in range(len(data)-1):
        if data[i][0] not in label_dict.keys():
            xyxy = "{},{},{},{}".format(int(data[i][1][0]),int(data[i][1][2]),int(data[i][1][1]),int(data[i][1][3]))
            label_dict[data[i][0]] = xyxy
        elif data[i][0] in label_dict.keys():
            xyxy = "{},{},{},{}".format(int(data[i][1][0]), int(data[i][1][2]), int(data[i][1][1]), int(data[i][1][3]))
            value = label_dict[data[i][0]] + '|' + xyxy
            label_dict[data[i][0]] = value
    im0 = data[-1]
    headers = {'Content-Type':'application/json;charset=UTF-8'}

    pingdao = save_path.split('/')[-1]
    for i in range(len(rtmp_url)):
        if pingdao in rtmp_url[i]:
            posturl = back_url[i]
            break
        else:
            continue
    # print('我是正确的url:',posturl)

    url = "http://119.3.185.115:9991/server_v4" # 供测试使用
    dataContent = []
    for key,value in label_dict.items():
        temp = {}
        temp["type"] = key
        temp["flag"] = "3"
        temp["pos"] = value
        dataContent.append(temp)
    backurl = "http://49.233.13.40:12077/WebAPI/api/aiReceiveData"

    if len(dataContent) < 1:
        return 0
    value = {"base64": frame2base64(im0),
             "applydate":str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),
             "rtmpurl": backurl,
             "backurl": backurl,
             "data":dataContent,
             "BoxCode": "001",
             "CameraCode": "002"

    }

    #param = json.dumps(value,cls=MyEncoder,indent=4)
    try:
        res = requests.post(url=backurl,json=value,headers=headers,timeout=0.15) # 正式使用时，应替换为posturl
    except:
        res = "超时"

    return "{}".format(res)

def detect(save_img=False):

    thread = Thread(target=thread_read_backurl, daemon=True)
    thread.start()
    time.sleep(1) #保证拿到的是更新后的rtmp_url和back_url

    # source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, False, opt.save_txt, opt.img_size

    save_img = not opt.nosave  # save inference images

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(  # isnumeric()方法检测字符串是否只由数字组成。
        ('rtsp://', 'rtmp://', 'http://'))


    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=32)


    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    names = [str(i) for i in range(1,24)]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    traj = 0

    while True: #模型初始化检测，如果rtmp_url 为空，则暂时静置模型
        if len(rtmp_url) > 0 and len(back_url) > 0:
            for path, img, im0s, vid_cap in dataset:
                try:
                    img.shape
                    image, detection = np.array(list(yolo.detect_image(img)), dtype=object)
                except:
                    continue
                data = []
                boxes = []
                if len(detection):
                    for i in range(len(detection)):
                        boxes.append(detection[i][:4])
                        detections = dict()
                        type = {"type": detection[i][-1]}
                        flag = {"flag": 3}
                        pos = {"pos": ','.join(str(i) for i in (detection[i][:4]))}

                        detections.update(type)
                        detections.update(flag)
                        detections.update(pos)
                        data.append(detections)
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                # pred 为 n行6列的数据
                # res = pred[0]
                # labels = pred[0][:,-1].int().tolist()
                #
                # if set([5,6]).issubset(labels) or set([7,8]).issubset(labels):
                #     xiyan = []
                #     shouji = []
                #
                #     for i in range(len(labels)):
                #         if labels[i] not in [5,6,7,8]:
                #             pass
                #         elif labels[i] == 5:
                #             xiyan.append([pred[i][:4]])
                #         elif labels[i] == 7:
                #             shouji.append([pred[i][:4]])
                #
                #
                #     for i in range(len(labels)):
                #         if labels[i] not in [6,8]:
                #             res[i] = pred[i]
                #         elif labels[i] == 6:
                #             x1,y1,x2,y2 = pred[i][:4]
                #             for i in range(len(xiyan)):
                #                 x_f = range(int(xiyan[i][0]),int(xiyan[i][2]))
                #                 y_f = range(int(xiyan[i][1]), int(xiyan[i][3]))
                #                 if int(x1) in x_f and int(x2) in x_f and int(y1) in y_f and int(x1) in y_f:
                #                     res[i] = torch.FloatTensor([0,0,0,0,0,0])
                #                     break
                #                 elif int(x1) in x_f and int(y1) in y_f:
                #                     res[i] = torch.FloatTensor([0, 0, 0, 0, 0, 0])
                #                     break
                #
                #         elif labels[i] == 8:
                #             x1, y1, x2, y2 = pred[i][:4]
                #             for i in range(len(shouji)):
                #                 x_f = range(int(shouji[i][0]), int(shouji[i][2]))
                #                 y_f = range(int(shouji[i][1]), int(shouji[i][3]))
                #                 if int(x1) in x_f and int(x2) in x_f and int(y1) in y_f and int(x1) in y_f:
                #                     res[i] = torch.FloatTensor([0, 0, 0, 0, 0, 0])
                #                     break
                #                 elif int(x1) in x_f and int(y1) in y_f:
                #                     res[i] = torch.FloatTensor([0, 0, 0, 0, 0, 0])
                #                     break
                # else:
                #     pass
                # pred = res



                #pred = non_max_suppression2(pred, opt.conf_thres, opt.iou_thres)
                t2 = time_synchronized()

                # Apply Classifier


                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    t4 = time.time()
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        # biaoqian_names = ['D_AQM','WD_AQM','C_FGY','WC_FGY','XIYAN1','XIYAN2','W_SJ1','W_SJ2','FY_WQ',
                        #                   'FY_HL','SDG_HG','SDG_WHG','JHP_HG','JHP_WHG1','JHP_WHG2','Huo','Huo_Y',
                        #                   'YC','W_DB','MaMian','LuJing','LieFeng','Qiang_KaiLie']
                        biaoqian_names = ['halmet','nohalmet','reflective_vest','no_reflective_vest','smoke','smoke',
                                          'play_cellphone','play_cellphone','climb','climb','sweeper','sweeper_wrong',
                                          'fire_basin','fire_basin_wrong','fire_basin_wrong','fire','fire','dust','no_base',
                                          'rough','exposed_tendons','crack']
                        post_res = []
                        # original = im0.copy()
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or (not view_img):  # Add bbox to image,保证post图片是画框的；
                                if names[int(cls)] in ['1','3','11','13']:
                                    continue
                                label = f'{biaoqian_names[int(cls)]}'
                                # label = f'{names[int(cls)]} {conf:.2f}'
                                post_res.append([label,xyxy])
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                # im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # post_res.append(original)
                        post_res.append(im0)
                        post_res = post(post_res,save_path)
                        if post_res == 0:
                            print("图片没有检测结果，因此不进行post")
                        elif "200" in post_res:
                            print("图片post成功")
                        else:
                            print("图片post失败")


                    # Stream results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    '''
                    cv2.IMWRITE_PNG_COMPRESSION
                    若想获得不同压缩程度保存的图片，这个参数对于cv2.imwrite很重要，使用方法：  
                    cv2.imwrite("img_dir", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    其中0代表图片保存时的压缩程度，有0-9这个范围的10个等级，数字越大表示压缩程度越高
                    '''
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        else:  # 'video' or 'stream'
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    # save_path += '.mp4'
                                    name = '_' + "%02d" % (traj/4) + '.jpg'  #这里假设有四个rtmp流，但是在实际过程中，应该是 len(rtmp_url)
                                    traj += 1
                                    cv2.imwrite(save_path+name, im0, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    t3 = time_synchronized()
                    # Print time (inference + NMS + POST + SAVE)
                    print(f'{s}Done. ({(t3 - t4) + float(t2 - t1)/len(pred):.3f}s)')
                # dataset(source)  # 监听函数
                time.sleep(5) # 睡眠1秒，等待datasets更新

            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                print(f"Results saved to {save_dir}{s}")

            print(f'Done. ({time.time() - t0:.3f}s)')

        else:
            time.sleep(5)
                                # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            # vid_writer.write(im0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./rtmp.txt', help='source')  # file/folder, 0 for webcam  /home/Oyj/yolov5/data/video /home/Oyj/datasetsCreated/remoteNetDatasets/data/test/images/
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default=False, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', default=True, help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='./runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
#    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
