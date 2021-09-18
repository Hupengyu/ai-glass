# -------------------------------------#
#       创建YOLO类
# -------------------------------------#
# "UTF-8"
import numpy as np
import torch
import torch.nn as nn
import configparser
from yolox.utils.boxes import yolo_correct_boxes
import cv2
import time
from yolox.utils import fuse_model, postprocess
import importlib
import os
import sys


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   都需要修改！
# --------------------------------------------#
def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    image = padded_img

    image = image.astype(np.float32)
    image = image[:, :, ::-1]
    image /= 255.0
    if mean is not None:
        image -= mean
    if std is not None:
        image /= std
    image = image.transpose(swap)
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image, r


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


class YOLO(object):
    _defaults = {
        "model_path": 'weights/best_ckpt.pth.tar',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "backbone": 'mobilenetv2',
        "model_image_size": (608, 608),
        "confidence": 0.25,
        "iou": 0.3,
        "cuda": False,
        'rgb_means': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'test_size': (608, 608),
        'confthre': 0.3,
        'nmsthre': 0.65,
        'depth': 1.00,
        'width': 1.00,
        'num_classes': 10,
        # 'name' : "yolox_s"
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__()

        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        # self.opt = self.make_parser().parse_args()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]

        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

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

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):

        exp = get_exp_by_name('yolox-s')

        model = exp.get_model()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        if self.cuda:
            # model.cuda()
            model.to(device)
        model.eval()
        ckpt_file = self.model_path
        ckpt = torch.load(ckpt_file, map_location=device)

        model.load_state_dict(ckpt["model"])
        self.net = fuse_model(model)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        config = configparser.ConfigParser()
        config.read(r'model_data/conf.ini', encoding='UTF-8')
        img, ratio = preproc(image, self.model_image_size, self.rgb_means, self.std)
        image_shape = np.array(np.shape(image)[0:2])
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        # img = torch.from_numpy(image).to(device)
        with torch.no_grad():
            outputs = self.net(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        """"
        [[     257.75      237.39      311.08      297.37     0.55826           3][271.53      202.19      302.19      235.04     0.51581           0]]
        """

        detecitions = []

        try:
            t3 = time.time()

            batch_detections = outputs[0].cpu().numpy()
            top_index = batch_detections[:, 4] > self.confthre

            top_label = np.array(batch_detections[top_index, -1], np.int32)
            top_bboxes = np.array(batch_detections[top_index, :4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
            # 去掉灰条
            boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                       np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)

            for i, c in enumerate(top_label):
                predicted_class = self.class_names[c]  # c = label
                top, left, bottom, right = boxes[i]

                # boxes 为一张图片的所有框
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5

                top = max(0, (np.floor(top + 0.5).astype('int32')))
                left = max(0, (np.floor(left + 0.5).astype('int32')))
                bottom = min(np.shape(image)[0], (np.floor(bottom + 0.5).astype('int32')))
                right = min(np.shape(image)[1], (np.floor(right + 0.5).astype('int32')))

                boxess = boxes.tolist()
                no_class = []
                noneed = config.get("pass", 'list')
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

                detection = [left, top, right, bottom, predicted_class_ch]
                detecitions.append(detection)

            return image, detecitions
        except:
            # image = np.array(image)
            return image, detecitions


if __name__ == '__main__':
    yolo = YOLO()
    im_path = '5005008401.jpg'
    img = cv2.imread(im_path)
    print(img)
    print(type(img))
    yolo.detect_image(img)
    """
    [[[127 127 127]
  [127 127 127]
  [127 127 127]
  ...
  [127 127 127]
  [127 127 127]
  [127 127 127]]

 [[127 127 127]
  [127 127 127]
  [127 127 127]
  ...
  [127 127 127]
  [127 127 127]
  [127 127 127]]

 [[127 127 127]
  [127 127 127]
  [127 127 127]
  ...
  [127 127 127]
  [127 127 127]
  [127 127 127]]

 ...

 [[127 127 127]
  [127 127 127]
  [127 127 127]
  ...
  [127 127 127]
  [127 127 127]
  [127 127 127]]

 [[127 127 127]
  [127 127 127]
  [127 127 127]
  ...
  [127 127 127]
  [127 127 127]
  [127 127 127]]

 [[127 127 127]
  [127 127 127]
  [127 127 127]
  ...
  [127 127 127]
  [127 127 127]
  [127 127 127]]]
<class 'numpy.ndarray'>
    """
