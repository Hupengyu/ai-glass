FROM 192.168.2.179:5000/jky/ai-glass:v2

ENV LANG=en_US.UTF-8 && LANGUAGE=en_US:en

MAINTAINER hupengyu

COPY requirements-base.txt /home/pro/JKY-AI-CX/

RUN pip install -r /home/pro/JKY-AI-CX/requirements-base.txt -i https://mirrors.aliyun.com/pypi/simple

ADD . /home/pro/JKY-AI-CX/

WORKDIR /home/pro/JKY-AI-CX/

RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

CMD ["start.sh"]