FROM python:3.6.13

MAINTAINER hupengyu

RUN apt update && apt-get -y install libgl1-mesa-glx && apt-get -y install vim

COPY requirements-base.txt /ai-glass/

RUN pip install -r /ai-glass/requirements-base.txt -i https://mirrors.aliyun.com/pypi/simple

ADD . /ai-glass

WORKDIR /ai-glass

RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

CMD ["run.sh"]