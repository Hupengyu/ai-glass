### nginx镜像版本
- nginx:latest
  - 初始版本
- nginx:v1
  - apt-get 源更新
  - vim
- nginx:v2
  - apt-get 源更新
  - vim 
  - 将所有的配置在docker镜像中写好
- nginx:v3[可能用不上]
  - apt-get 源更新
  - vim 
  - 挂载目录到宿主机，以便后续更改
    - docker run -d -p 8081:80 -v /data/nginx/html:/usr/share/nginx/html -v /data/nginx/nginx.conf:/etc/nginx/nginx.conf  -v /data/nginx/logs:/var/log/nginx -v /data/nginx/conf.d:/etc/nginx/conf.d nginx:v1

- 执行
  - docker run -itd --restart=always -p 8081:81 192.168.2.179:5000/jky/nginx:v1 /bin/bash
  - docker run -d --restart=always -p 8081:81 192.168.2.179:5000/jky/nginx:v1[运行时]

### nginx-rtmp镜像版本
- alfg/nginx-rtmp:latest
- 运行
  - docker run -itd --restart=always -p 8777:1935 --name nginx-rtmp alfg/nginx-rtmp

### 流媒体服务器
- 推流实例
  - ffmpeg -re -i 1.mp4 -f flv rtmp://119.3.185.115:8777/stream/123
  - ffplay rtmp://119.3.185.115:8777/stream/123
- 推流监控器
  - ffmpeg -i "rtsp://admin:jky123456@192.168.2.188" -vcodec copy -acodec aac -ar 44100 -strict -2 -ac 1 -f flv -s 4000x3000 -q 10 -f flv "rtmp://119.3.185.115:8777/stream/123"
- 查看
  - ps -ef |grep nginx |grep -v grep
