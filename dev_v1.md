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

### 从私有仓库拉取镜像
- 查看仓库中的镜像
- curl -XGET http://192.168.2.179:5000/v2/_catalog
  - {"repositories":["jky/ai-glass"]}      #显示上传成功
- 查看镜像tag
  - curl http://192.168.2.179:5000/v2/jky/ai-glass/tags/list
    - {"name":"jky/ai-glass","tags":["v1"]}
- 拉取镜像
  - docker pull 192.168.2.179:5000/jky/ai-glass:v1

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
  - docker run -it --name ai-glass --privileged=true --gpus all --restart=always -p 3306:3306 192.168.2.179:5000/jky/ai-glass:v1 bin/bash

- 修改容器的编码格式：
  - export LANGUAGE=zh_CN.UTF-8
  - export LANG=zh_CN.UTF-8

****

### 生成镜像
- docker commit -a "jky/ai-glass/hpy" a16b50cb622d jky/ai-glass:v1

### Dockerfile
- 根据本地镜像生成docker基础镜像