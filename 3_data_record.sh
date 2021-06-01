#! /bin/bash
# start avoid model
cd ~/ros_ws/src/avoid_model_port/scripts

source ~/venv/torch/bin/activate
python raw_record.py --source 4 --fps 30 --cam_type 3