#!/bin/bash
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

python rtmp2img.py &
python server/jky_serverx_camera.py &
python video_c.py