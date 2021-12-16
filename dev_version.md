# ai-glass项目docker images及运行文档 

### ai-glass镜像版本
- 192.168.2.179:5000/jky/ai-glass:v1(纯净版)
  - ai code

- 192.168.2.179:5000/jky/ai-glass:v2(做镜像用)
  - v2中还有一些其他的包，但是以此位基础镜像可以了

- 192.168.2.179:5000/jky/ai-glass:v3(docker-host)
  - 该镜像是基于v2的镜像通过Dockerfile生成的，包含最新代码 
  - docker run -it --privileged=true --gpus all --restart=always --net=host 192.168.2.179:5000/jky/ai-glass:v3

- 192.168.2.179:5000/jky/ai-glass:v4(2021-1-1版本)
  - 使用v2作为基础镜像
  - project：JKY-AI-BOX
  - 内网穿透：sunny-ngrok 
  - 运行[使用本机的网络：--net=host]
    - docker run -it --privileged=true --gpus all --restart=always --net=host 192.168.2.179:5000/jky/ai-glass:v4
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
      - RUN chmod 777 start.sh
      - CMD ["sh","start.sh"]