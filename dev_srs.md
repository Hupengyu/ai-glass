# srs外网服务器部署
- 119.3.185.115
- ZJXX@2021.COM

### srs目录文件挂载

- docker run -d --restart=always -p 8777:1935 -v /home/docker/srs/conf/:/usr/local/srs/conf/ -v /home/docker/srs/objs/:/usr/local/srs/objs/ ossrs/srs:latest
