#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
#"UTF-8"
import numpy as np
import torch
import torch.nn as nn
import configparser
from yolox.utils import fuse_model, postprocess
import importlib
pre = []
image_path = {}
import sys
sys.path.append('..')
from hash_rmrept import hash_rmrept
import cv2
import numpy as np
import base64
import requests
import torch.multiprocessing
from multiprocessing import Value
import time
pid_info_path = './model_data/Project_pid.txt'
result_paths = 'result'
fixid = './model_data/BoxStatu.txt'
alive = Value('b', False)
source = './model_data/url_cameras.txt'
import os
#from retinaface import Retinaface
#retinaface = Retinaface()
from rtmp2img2 import LoadStreams
url_lss = []
status = False
stop = False
images = {}
predict = {}


def base_img(img_im):
    img_im = np.array(img_im)

    return base64.b64encode(cv2.imencode('.jpg',img_im)[1]).decode()
#去重
def image_rmrept(frame,img):
    #reba = face_rmrept(frame, danger_list, face_path, pic_path)

    res = hash_rmrept(frame, img)
    if res == 'same_pic':

        return True
    elif res == 'diff_pic':
        return False

def post(fras,data,rtmpurl):
    with open(fixid, 'r') as f:
        fixs = f.readlines()
        for fix in fixs:
            if fix is None:
                break
            else:
                ffix = fix.split(',')
                ProjectID = ffix[0]
                BoxCode = ffix[1]
                backurl = ffix[2]
    #data = data.split('')
    headers = {'Content-Type':'application/json;charset=UTF-8'}
    value = {"base64": base_img(fras),
             "checkdate": str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),
             "rtmpurl": rtmpurl,
             "data": data,
             "BoxCode": BoxCode,

             "ProjectID": ProjectID,

             }
    t2 = time.time()
    #print(value)
    res = requests.post(url=backurl,json=value,headers=headers)
    print(res.text)
    # except:
def multi_preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    res = []
    gns = []
    for i in range(len(image)):
        tmp = image[i].copy()
        if len(tmp.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(tmp)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        tmp = padded_img

        tmp = tmp.astype(np.float32)
        tmp = tmp[:, :, ::-1]
        tmp /= 255.0
        if mean is not None:
            tmp -= mean
        if std is not None:
            tmp /= std
        tmp = tmp.transpose(swap)
        tmp = np.ascontiguousarray(tmp, dtype=np.float32)
        res.append(tmp)
        gns.append(r)
    return res, gns


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp

def get_exp_by_name(exp_name):
    import yolox
    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    filedict = {
        "yolox-s": "yolox_s.py",
    }
    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]

        cls_id = int(cls_ids[i])
        if cls_id ==0:
            continue

        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


VOC_CLASSES = ('1','2','3','4','5','6','7','8','9','10','11','12',)
class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/best.pth.tar',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/voc_classes.txt',
        "backbone"          : 'mobilenetv2',
        "model_image_size"  : (608, 608),
        "confidence"        : 0.25,
        "iou"               : 0.3,
        "cuda"              : True,
        'rgb_means'         :(0.485, 0.456, 0.406),
        'std'               :(0.229, 0.224, 0.225),
        'test_size'         :(608, 608),
        'confthre'          :0.3,
        'nmsthre'           :0.65,
        'depth'             :1.00,
        'width'             :1.00,
        'num_classes'       :12,
        #'name' : "yolox_s"
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__()

        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        #self.opt = self.make_parser().parse_args()
        self.generate()
        self.device = 'gpu'

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]

        return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def multi_visual(self,output, img, cls_conf=0.35):
        res = []
        gns = []

        # logger.info(f'output length is {len(output)} ')
        # logger.info(f'img length is {len(img)} ')
        for i in range(len(img)):
            # logger.info(f'img[i] is {img[i].shape}')
            image = img[i].copy()
            # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            gns.append(torch.Tensor(image.shape)[[1, 0, 1, 0]])
            # logger.info(f'img[i] is {image.dtype}')
            if output[i] is None:
                res.append(image)
                continue

            ratio = min(self.test_size[0] / image.shape[0], self.test_size[1] / image.shape[1])

            out = output[i].cpu()

            bboxes = out[:, 0:4]

            # preprocessing: resize
            bboxes /= ratio

            cls = out[:, 6]
            scores = out[:, 4] * out[:, 5]

            vis_res = vis(image, bboxes, scores, cls, cls_conf, self.class_names)
            # logger.info(f'img[i] is {vis_res.shape}')

            res.append(vis_res)

        return res, gns
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):


        exp = get_exp_by_name('yolox-s')

        model = exp.get_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.cuda:
            model.cuda()
        model.eval()
        ckpt_file =self.model_path
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model"])
        self.net = fuse_model(model)

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_sets,image,paths):
        global images
        config = configparser.ConfigParser()
        config.read(r'model_data/conf.ini', encoding='UTF-8')
        #1080 * 1920 *3
        image_sets, ratio = multi_preproc(image_sets, self.model_image_size, self.rgb_means, self.std)
        #3*608*608 图片转比例
        for i in range(len(image_sets)):
            image_sets[i] = torch.from_numpy(image_sets[i]).unsqueeze(0) #转tensor
            # logger.info('shape {}'.format(img[i].shape))
        if len(image_sets) == 1:
            img = image_sets[0]
        else:
            tmp = image_sets[0]
            for i in range(1,len(image_sets)):
                tmp = torch.cat((tmp, image_sets[i]), 0)
            img = tmp

        img = img.cuda()  #[3,3,608,608] tensor
        with torch.no_grad():
            outputs = self.net(img)#[3,3,608,608]
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

         # 去重临时字典
        single_image_data = {}  # 单张图片的信息
        one_batch_images_result = [] # 本批次结果
        # 处理本批次的图片
        for i in range(len(image)): # 遍历每一张图片
            if outputs[i] is None:
                continue
            path = paths[i]
            ima = image[i].copy()
            if ima is None:
                continue
            batch_detections = outputs[i].cpu().numpy()
            top_index = batch_detections[:,4] > self.confthre

            top_label = np.array(batch_detections[top_index,-1],np.int32)
            bboxes = batch_detections[:, 0:4]
            bboxes /= ratio[i]
            cls = batch_detections[:, 6]
            scores = batch_detections[:, 4] * batch_detections[:, 5]
            data = []  # 每张图片的type信息
            predict = []

            for i, c in enumerate(top_label):
                # if scores[i]<0.45:
                #     continue

                predicted_class = self.class_names[c] #c = label
                box = bboxes[i]
                no_class = []

                noneed = config.get("pass",'list')
                if noneed is None:
                    continue
                else:
                    noneed = noneed.split(',')

                for i in range(len(noneed)):
                    a = noneed[i]
                    no_class.append(a)
                if predicted_class in no_class:
                    continue
                else:
                    predicted_class_ch = config.get("config", predicted_class)

                detection = ','.join(str(int(i)) for i in box)
                predict.append(predicted_class_ch)

                detections = dict()
                type = {"type": predicted_class_ch}
                flag = {"flag": 3}
                pos = {"pos": str(detection)}

                detections.update(type)
                detections.update(flag)
                detections.update(pos)
                data.append(detections) #[{'type': '1005,238,1077,328', 'flag': 3, 'pos': 'no_reflective_vest'}， []， []]

            if data is None:
                continue
            # 存储每张图片的信息
            single_image_data_value = []
            single_image_data_value.append(ima)
            single_image_data_value.append(data)
            single_image_data_value.append(predict)

            single_image_data[path] = single_image_data_value
        #one_batch_images_result.append(single_image_data)
        return single_image_data



def to_work(dir_l):
    global rtmp_url,rta,source,stop,images,predict

    pid = os.getpid()
    rta = False


    while True:
        if stop:
            print("jinchengjieshu")
            break
        else:
            dataset = LoadStreams(dir_l, img_size=608, stride=32)
            imgdict = []
            i = 0
            for path, img, im0s, vid_cap in dataset:
                image_sets = im0s.copy()
                # time.sleep(1) # 测试降速
                i = i+1
                result  = YOLO().detect_image(image_sets,im0s,path)
                path_ = result.keys()
                for i,j in enumerate(path_):
                    pre_ = result.get(j)[-1]
                    img_ = result.get(j)[0]
                    #img_ =
                  #  img_re = retinaface.detect_image(img_)
                    img_re = img_

                    data_ = result.get(j)[1]
                    if len(data_):

                    #data_= data_.split(';')[1]
                        if j not in predict.keys():

                            post(img_re, data_, j)
                            print(data_)
                            predict[j] = pre_
                            images[j] = img_re
                        else:
                            result_img = image_rmrept(images.get(j), img_re)
                            if pre_ == predict.get(j) and result_img:
                                predict[j] = pre_
                                images[j] = img_re
                                continue
                            else:

                                post(img_re, data_, j)
                                print(data_)
                                images[j] = img_re
                                predict[j] = pre_



                          # 同一视频流下不同时刻的图片
                              # 视频流去重


            'datas.append([ima,data,predict,path])'
    os.system('kill -9 %s' % pid)

def read_account(filename):
    with open(filename, 'r+', encoding='utf-8') as f:
        res = f.readlines()
        f.seek(0)
        f.truncate()
        f.close()
   #
def main():
    # 创建子进i程
    global path,url_lss,status,stop
    #read_account(source)

    while True:

        with open(source, 'r', encoding='utf-8') as f_r:
            dirs = f_r.readlines()
            dirs = [x.strip('\n') for x in dirs]    #  暂存txt-list

            if len(dirs) > 0:   # txt非空
                if len(url_lss) == 0:   # 第一次为url_lss赋值
                    for dir in dirs:    # 将txt内容存入url_lss
                        url_lss.append(dir)
                    status = True
                    stop = True
                else:   # 非第一次为url_lss赋值
                    if dirs != url_lss: # url_lss与txt-list不等
                        url_lss.clear() # 清空url_lss
                        for dir in dirs:    # 重新为url_lss赋值
                            url_lss.append(dir)
                        status = True
                        stop = True

            f_r.close()

        if status:
            rtmp_num = len(url_lss) # url_lss-list
            if rtmp_num <= 0 :
                continue
            num_dir = rtmp_num//5.1+1 # 5的倍数
            url_lss_copy = url_lss.copy()
            while num_dir > 0:    # 循环读取两部分
                #ran_num = (rtmp_num//11)+1
                rtmp_num_tmp = len(url_lss_copy)  # url_lss-list
                if rtmp_num_tmp<=5:
                    ctx = torch.multiprocessing.get_context('spawn')
                    # ctx = multiprocessing.get_context('spawn')
                    print("开始检测1")
                    son_p1 = ctx.Process(target=to_work, args=([url_lss_copy]), daemon=False)
                    son_p1.start()
                    status = False
                    num_dir = num_dir - 1
                else:
                    # 删除前5个
                    start_url = []
                    for i,j in enumerate(url_lss_copy):
                        if i <5:
                            url_lss_copy.remove(j)

                            start_url.append(j)
                        # i += 1
                    ctx = torch.multiprocessing.get_context('spawn')
                    # ctx = multiprocessing.get_context('spawn')
                    print("开始检测2")
                    son_p1 = ctx.Process(target=to_work, args=([start_url]), daemon=False)
                    son_p1.start()
                    status = False
                    num_dir = num_dir - 1

if __name__ == '__main__':
    main()
    # dataset = LoadStreams(dir_l, img_size=608, stride=32)
    # print(dataset)
