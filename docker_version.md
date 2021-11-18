# docker images 

### ai-glass镜像版本
- 192.168.2.179:5000/jky/ai-glass:v1(纯净版)
  - ai code
- 192.168.2.179:5000/jky/ai-glass:v3(docker-host)
  - docker run -it --privileged=true --gpus all --restart=always --net=host 192.168.2.179:5000/jky/ai-glass:v3