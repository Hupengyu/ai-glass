# srs外网服务器部署
- 119.3.185.115
- ZJXX@2021.COM

### srs配置文件
https://blog.csdn.net/u011374856/article/details/107332309


### srs目录文件挂载

- docker run -d --restart=always -p 8777:1935 -v /home/docker/srs/conf/:/usr/local/srs/conf/ -v /home/docker/srs/objs/:/usr/local/srs/objs/ ossrs/srs:latest
