# srs外网服务器部署

### srs流媒体服务器说明
- srs流媒体服务器以docker容器的方式部署
- tx2从工地接收视频流并经过隐患识别，然后将识别后的视频流通过网络上传至srs
- 其他服务向srs请求视频流

### 流媒体服务器部署地址
- 119.3.185.115
- ZJXX@2021.COM

### srs镜像版本
- ossrs/srs:latest
  - 先运行docker容器，然后cp出配置文件

### srs目录文件挂载
- 获取srs的docker镜像L:ossrs/srs,并将其中的配置文件cp出宿主机
  - https://blog.csdn.net/u011374856/article/details/107332309/
- 修改配置文件，实现低延迟
  - https://blog.csdn.net/u011374856/article/details/107332309
- 将配置文件挂载处理，并重新运行容器
  - 原配置
    - docker run -d --restart=always -p 8777:1935 -v /home/docker/srs/conf/:/usr/local/srs/conf/ -v /home/docker/srs/objs/:/usr/local/srs/objs/ ossrs/srs:latest
  - 低延迟配置
    - docker run -d --restart=always -p 8777:1935 -v /home/docker/srs/conf/:/usr/local/srs/conf/ -v /home/docker/srs/objs/:/usr/local/srs/objs/ ossrs/srs:latest ./objs/srs -c conf/realtime.conf

### 流媒体服务器推拉流
- 推流
  - ffmpeg -i "rtsp://admin:jky123456@192.168.2.188" -vcodec copy -acodec aac -ar 44100 -strict -2 -ac 1 -f flv -s 4000x3000 -q 10 -f flv "rtmp://119.3.185.115:8777/stream/rtsp://admin:jky123456@192.168.2.188"
- 拉流
  - 使用VLC
    - rtmp://119.3.185.115:8777/stream/rtsp://admin:jky123456@192.168.2.188