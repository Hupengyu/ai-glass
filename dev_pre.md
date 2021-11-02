### client端，安装mysql
- 


### client端，安装docker,docker-run-time
- 安装docker
- 安装nvidia-container-runtime
    - sudo apt-get install nvidia-container-runtime -y
    - systemctl stop docker
    - dockerd --add-runtime=nvidia=/usr/bin/nvidia-container-runtime
    - 测试方式：进入一个容器，使用--gpus=all命令运行一个容器，并nvidia-smi,则成功
    
### 获取server端镜像
- 1.查看服务器上的镜像：
    - curl 192.168.2.179:5000/v2/_catalog
- 2.从服务器拉取镜像
    - 修改配置文件
        - vi /etc/docker/daemon.json
    - 添加内容
        - {"insecure-registries":["192.168.2.179:5000"]}
    - 拉取服务器上的镜像
        - docker pull 192.168.2.179:5000/jky-ai/ai-glass:1.0


### 运行容器(gpu版，需要nvidia-container-runtime)
- 文件准备
    - weights
    - model_data
- docker run -it --gpus=all -v /home/project/oyj/JKY-AI-CX-weights:/ai-glass/weights -v /home/project/oyj/JKY-AI-CX-model_data:/ai-glass/model_data jky-ai/ai-glass:1.0 /bin/bash

### 运行程序(已写好)
- 在Dockerfile中使用CMD运行run.sh
CMD ["run.sh"]

- 手动运行
sh run.sh
