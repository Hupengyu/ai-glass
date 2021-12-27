## 内网穿透ngrok

### 功能说明
- 实现内网穿透功能: 让外网可以进入盒子所在的局域网

### Sunny-ngrok端口
- 赠送域名:http://jky.gz2vip.91tunnel.com
- 本地端口:192.168.2.194:8666rtsp://admin:jky123456@192.168.2.188
- 隧道id:155825333161

### 运行
- JKY-AI-BOX容器内部运行
  - 下载sunny可执行文件
    - https://www.ngrok.cc/download.html
    - 放入JKY-AI-BOX目录下
  - 加入start.sh脚本运行命令: setsid ./sunny clientid 155825333161