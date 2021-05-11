#! /bin/bash
# start avoid model

cd ~/ros_ws/src/avoid_model_port/scripts

source ~/venv/torch/bin/activate
python detect.py --webcam --source 4 --cam_freq 2 --visual -score 0.6

