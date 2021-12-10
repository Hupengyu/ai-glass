# docker images 

### ai-glass镜像版本
- 192.168.2.179:5000/jky/ai-glass:v1(纯净版)
  - ai code
- 192.168.2.179:5000/jky/ai-glass:v2(做镜像用)
  - v2中还有一些其他的包，但是以此位基础镜像可以了
- 192.168.2.179:5000/jky/ai-glass:v3(docker-host)
  - 该镜像是基于v2的镜像通过Dockerfile生成的，包含最新代码 
  - docker run -it --privileged=true --gpus all --restart=always --net=host 192.168.2.179:5000/jky/ai-glass:v3
- 192.168.2.179:5000/jky/ai-glass:v4
  - name: JKY-AI-BOX 
  - 添加sunny-ngrok 
  - docker run -it --privileged=true --gpus all --restart=always --net=host 192.168.2.179:5000/jky/ai-glass:v4
  - 备注：需要在Dockerfile中添加CMD[‘start.sh’]