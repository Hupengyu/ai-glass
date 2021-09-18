import torch
import io

path = '/home/project/oyj/YOLOX-main/yolox/weights/yolox_jit.pth'
model = torch.jit.load(path, map_location=torch.device('cuda'))

x = torch.rand(10,3,608,608).cuda()
pre = model(x)
print(pre.shape)
