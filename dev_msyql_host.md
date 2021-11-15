### JETSON-AGX-XAVIER 配置
- cat /proc/version
  - Linux version 4.9.201-tegra (buildbrain@mobile-u64-4551) 
  - (gcc version 7.3.1 20180425 [linaro-7.3-2018.05 revision d29120a424ecfbc167ef90065c0eeb7f91977701] (Linaro GCC 7.3-2018.05) )
- uname -a
  - Linux j 4.9.201-tegra 
  - aarch64 GNU/Linux
- sb_release -a
  - No LSB modules are available.
  - Distributor ID:	Ubuntu
  - Description: Ubuntu 18.04.6 LTS
  - Release: 18.04
  - Codename: bionic  //ubuntu的代号名称

### jetson镜像
- JetPack 4.5 (L4T R32.5.0)
  - **l4t-ml:r32.5.0-py3**
    - TensorFlow 1.15[删除]
    - PyTorch v1.7.0
    - torchvision v0.8.0
    - torchaudio v0.7.0
    - onnx 1.8.0
    - CuPy 8.0.0
    - numpy 1.19.4
    - numba 0.52.0
    - OpenCV 4.1.1
    - pandas 1.1.5
    - scipy 1.5.4
    - scikit-learn 0.23.2[删除]
    - JupyterLab 2.2.9[删除]
- 由git拉取镜像命令
  - sudo docker pull nvcr.io/nvidia/l4t-ml:r32.5.0-py3

- 删除无用pip包
  - TensorFlow 1.15[删除]
  - JupyterLab 2.2.9[删除]
  - scikit-learn 0.23.2[删除]
  - ...
- 当前包
  - absl-py (0.11.0)
  - appdirs (1.4.4)
  - argon2-cffi (20.1.0)
  - astor (0.8.1)
  - async-generator (1.10)
  - attrs (20.3.0)
  - certifi (2020.12.5)
  - cffi (1.14.4)
  - chardet (4.0.0)
  - cupy (8.0.0b4)
  - cycler (0.10.0)
  - Cython (0.29.21)
  - dataclasses (0.8)
  - decorator (4.4.2)
  - defusedxml (0.6.0)
  - entrypoints (0.3)
  - fastrlock (0.5)
  - future (0.18.2)
  - futures (3.1.1)
  - gast (0.2.2)
  - h5py (2.10.0)
  - importlib-metadata (3.1.1)
  - ipykernel (5.4.2)
  - ipython (7.16.1)
  - ipython-genutils (0.2.0)
  - ipywidgets (7.5.1)
  - json5 (0.9.5)
  - jsonschema (3.2.0)
  - matplotlib (2.1.1)
  - numba (0.52.0)
  - numpy (1.19.4)
  - onnx (1.8.0)
  - packaging (20.8)
  - pandas (1.1.5)
  - Pillow (8.0.1)
  - pip (9.0.1)
  - protobuf (3.14.0)
  - pycuda (2020.1)
  - Pygments (2.2.0)
  - pyparsing (2.2.0)
  - pytools (2020.4.4)
  - pytz (2018.3)
  - PyYAML (3.12)
  - requests (2.25.1)
  - six (1.11.0)
  - scipy (1.5.4)
  - setuptools (51.0.0)
  - torch (1.7.0)
  - torchaudio (0.7.0a0+ac17b64)
  - torchvision (0.8.0a0+45f960c)
  - urllib3 (1.26.2)
  - wcwidth (0.2.5)
  - webencodings (0.5.1)
  - Werkzeug (1.0.1)
  - wheel (0.36.1)
  - widgetsnbextension (3.5.1)
  - wrapt (1.12.1)
  - zipp (3.4.0)
- Jetson AGX Xavier更换apt-get源
  - cp /etc/apt/sources.list /etc/apt/sources.list.bak
  - echo deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-updates main restricted universe multiverse  >> sources.list
  - echo deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-security main restricted universe multiverse >> sources.list
  - echo deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-backports main restricted universe multiverse >> sources.list
  - echo deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic main universe restricted >> sources.list
  - apt-get update
- 安装vim/gedit
  - apt-get install vim -y
  - apt-get install gedit -y
- pip3源修改
    - mkdir ~/.pip
    - vim ~/.pip/pip.conf
      - [global]
      - index-url = https://mirrors.aliyun.com/pypi/simple/
      - index-url = https://pypi.tuna.tsinghua.edu.cn/simple/[备选]
- 添加包
  - flask
  - idna
  - typing_extensions
  - loguru
  - thop


### 配置docker宿主机的mysql环境
- 安装mysql(默认会使用)
  - apt-get update
  - apt-get install mysql-server
- 启动mysql
  - service mysql status
  - service mysql start
- 创建mysql数据库
  - create database face_feature_data;  
- 创建数据表
  - create table face_feature_embedding(name VARCHAR(25), embedding VARCHAR(3000));

### docker 容器自启动
- docker服务能随OS启动而启动
  - sudo systemctl enable docker.service    # 设置开机启动
  - sudo systemctl list-unit-files | grep enable    # 查看是否设置开机启动
  - sudo systemctl list-units --type=service    # 查看已启动的服务
- docker容器能随docker服务启动而启动
  - --restart=always
- docker容器内的服务能随docker容器启动而启动

### 从私有仓库拉取镜像
- 查看仓库中的镜像
- curl -XGET http://192.168.2.179:5000/v2/_catalog
  - {"repositories":["jky/ai-glass"]}      #显示上传成功
- 查看镜像tag
  - curl http://192.168.2.179:5000/v2/jky/ai-glass/tags/list
    - {"name":"jky/ai-glass","tags":["v1"]}
- 从服务器拉取镜像
  - 修改配置文件
      - vim /etc/docker/daemon.json
  - 设置docker仓库权限
    - {
      "registry-mirrors": [
            "https://registry.docker-cn.com",
            "http://hub-mirror.c.163.com",
            "https://pee6w651.mirror.aliyuncs.com",
            "https://docker.mirrors.ustc.edu.cn"],
      "insecure-registries": ["registry-1.docker.io","192.168.2.179:5000"]
    }
  - 拉取服务器上的镜像
      - docker pull 192.168.2.179:5000/jky/ai-glass:v2

### 运行容器
- 挂载
  - model_data
  - code
- 配置
  - 读取硬件: --privileged=true
  - gpus: all
  - 开机自启动: --restart=always
  - 端口映射: 3306:3306
- 关闭mysql
  - killall mysqld[开机执行]
- 挂载Mysql配置
  - -v /opt/mysql_docker/mysql:/etc/mysql    挂载配置文件
  - -v /opt/mysql_docker/data:/var/lib/mysql  挂载数据文件 持久化到主机
  - -v /opt/mysql_docker/mysql/conf:/etc/mysql/conf.d  

- 运行
  - docker run -it --privileged=true --gpus all --restart=always 192.168.2.179:5000/jky/ai-glass:latest

- 修改容器的编码格式：
  - export LANG=en_US.UTF-8[如果有 en_US.utf8 优先使用]
  - export LANGUAGE=en_US:en
- 永久设置需在Dockerfile中设置环境字符集环境变量
  - ENV  LANG="en_US.UTF-8"


****
# 附录
### 生成镜像
- docker commit -a "jky/ai-glass/hpy" a16b50cb622d jky/ai-glass:v1

### 上传镜像
- docker push 192.168.2.179:5000/jky/ai-glass:v2

### Dockerfile
- 根据本地镜像生成docker基础镜像

### Docker镜像version
- 192.168.2.179:5000/jky/ai-glass:v1
  - 带项目
  - 代码未修改
  - 没有启动程序start.sh
  - 有环境有pip包
- 192.168.2.179:5000/jky/ai-glass:v2
  - 有启动程序
  - 可自启动
  - 可以运行的版本
- 192.168.2.179:5000/jky/ai-glass：1.0
  - apt-get
  - vim
  - code
  - LANG=en_US.UTF-8
  - pip3源修改
    - mkdir ~/.pip
    - vim ~/.pip/pip.conf
      - [global]
      - index-url = https://mirrors.aliyun.com/pypi/simple/
      - index-url = https://pypi.tuna.tsinghua.edu.cn/simple/[备选]