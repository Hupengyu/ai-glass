## 创建1.0 cuda镜像

###  1.根据nvidia/cuda-11.0-base镜像创建container
- docker run -it -v /home/project/oyj/JKY-AI-CX/:/ai-glass nvidia/cuda:11.0-base /bin/bash

###  2.将项目copy到container中
- docker cp /home/project/oyj/ai-glass/ image-name:1.0

###  3.在容器中下载所有需要的包
- 安装基本工具
    - apt update
    - apt-get -y install libgl1-mesa-glx
    - apt-get install libglib2.0-dev -y
    - apt-get -y install vim
    - apt-get install wget -y
    - 安装python(https://blog.csdn.net/weixin_44129085/article/details/104793114)
    - 设置python/python3的软连接(https://blog.csdn.net/maizousidemao/article/details/102810681;
                                https://www.jianshu.com/p/9d3033d1b26f)——————————nvidia/cuda镜像系统默认为python3.8
    - pip安装
        - wget https://bootstrap.pypa.io/get-pip.py
        - python get-pip.py
        - hash -r
    - 设置pip国内源
        - mkdir ~/.pip
        - vim .pip/pip.conf
        - [global]
          index-url = https://pypi.tuna.tsinghua.edu.cn/simple
          trusted-host = pypi.tuna.tsinghua.edu.cn
- 安装依赖包
    - pip install -r requirements-base.txt
- 生成镜像
    - docker commit dc26f65db72e ai_glass_cuda11.0:1.0
    
###  4.根据新生成的ai_glass_cuda11.0:1.0镜像生成container
- docker run -it -v /home/project/oyj/JKY-AI-CX-weights:/ai-glass/weights -v /home/project/oyj/JKY-AI-CX-model_data:/ai-glass/model_data ai_glass_cuda11.0:1.0 /bin/bash
