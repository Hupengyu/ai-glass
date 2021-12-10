## 内网穿透ngrok

### Sunny-ngrok

### docker镜像运行
- docker run -it --privileged=true --gpus all --restart=always --net=host jky/ai-glass/ngrok-test:1.0
- 内部运行
  - sunny-ngrok: setsid ./sunny clientid 155825333161