#! /bin/bash
# start avoid model

cd ~/ros_ws/src/avoid_model_port/scripts

source ~/venv/torch/bin/activate
python detect.py --webcam --source 4 --fps 2 --BM --visual --score 0.6 --cam_type 3
