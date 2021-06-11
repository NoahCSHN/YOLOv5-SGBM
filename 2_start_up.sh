#! /bin/bash
# start avoid model

cd ~/ros_ws/src/avoid_model_port/scripts

source ~/venv/torch/bin/activate
python detect.py --visual --webcam --source 4 --cam_type 3 --sm_numdi 43 --cam_type 3 --sm_UniRa 5 --sm_mindi -5 --sm_block 9 --sm_tt 10 --sm_pfc 63 --sm_sws 100 --sm_sr 2 --score 0.6 --sm_d12md 1