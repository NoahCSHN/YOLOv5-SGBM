#! /bin/bash
# start avoid model
cd ~/ros_ws/src/avoid_model_port/scripts

source ~/venv/cp37/bin/activate
python raw_record.py --source 4 --fps 4 --cam_type 3