# ai-glass项目文档 

### 项目说明
- 隐患识别环境准备
  - <tx2盒子>
    - 隐患识别代码主项目(JKY-AI-BOX)
    - <ngrok内网穿透服务>:实现外网访问请求
  - web服务器
    - <nginx流媒体服务器>:实现视频流的存储和转发
- 项目流程
  - 使用海视康威摄像头，通过局域网路由器交互视频流
  - 使用<tx2盒子>连接局域网获取视频流
  - 外网请求获取隐患识别视频(通过ngrok)
  - 运行隐患识别代码
    - 完成隐患识别并将边框画在视频流上，形成新的视频流
  - 将视频流上传至外网服务器
    - 使用<nginx流媒体服务器>接收视频流，并存储视频流
  - web端从<nginx流媒体服务器>获取隐患识别后的视频流

### ai-glass镜像版本
- 192.168.2.179:5000/jky/ai-glass:v1(纯净版)
  - ai code

- 192.168.2.179:5000/jky/ai-glass:v2(做镜像用)
  - v2中还有一些其他的包，但是以此位基础镜像可以了

- 192.168.2.179:5000/jky/ai-glass:v3(docker-host)
  - 该镜像是基于v2的镜像通过Dockerfile生成的，包含最新代码 
  - docker run -it --privileged=true --gpus all --restart=always --net=host 192.168.2.179:5000/jky/ai-glass:v3

- 192.168.2.179:5000/jky/ai-glass:v5(2022-1-1版本)
  - 时间同步:RUN /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone
  - 编码格式:RUN export LANG=en_US.UTF-8 ENV LANG=en_US.UTF-8
  - 推流ffmpeg:RUN apt-get install ffmpeg -y
  - 基础镜像:使用v2作为基础镜像
  - project名称：JKY-AI-BOX
  - 内网穿透sunny-ngrok:setsid ./sunny clientid 155825333161 
  - 运行[使用本机的网络：--net=host]
    - docker run -itd --privileged=true --gpus all --restart=always --net=host 192.168.2.179:5000/jky/ai-glass:v5 
  - 自启动docker容器内的项目：
    - Dockerfile中添加CMD[‘start.sh’]
  - 代码更新
    - 项目的更新在agx内host中，所有的代码在其中编写完成后再生产镜像，并做成容器
  - 自启动
    - docker服务能随OS启动而启动
      - sudo systemctl enable docker.service    # 设置开机启动
      - sudo systemctl list-unit-files | grep enable    # 查看是否设置开机启动
      - sudo systemctl list-units --type=service    # 查看已启动的服务
    - docker容器能随docker服务启动而启动
      - --restart=always
    - docker容器内的服务能随docker容器启动而启动
      - RUN chmod 777 start.sh(没有也行好像)
      - CMD ["sh","start.sh"x`