#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : 录数据
@Date     :2021/05/31 15:05:22
@Author      :Yyc
@version      :1.0
'''
import os,logging,sys,argparse,time,math,queue
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from threading import Thread
from utils.img_preprocess import Image_Rectification
from utils.Stereo_Application import Stereo_Matching,disparity_centre
from utils.dataset import DATASET_NAMES,loadfiles,loadcam
from utils.stereoconfig import stereoCamera
from utils.general import confirm_dir,timethis,timeblock,socket_client,calib_type,camera_mode,plot_one_box


#%%
def main():
    dataset = loadcam(pipe=args.source,cam_freq=args.fps,img_size=args.img_size,save_path=args.save_path,debug=args.debug,cam_mode=args.cam_type)
    file_path = confirm_dir(args.save_path,datetime.now().strftime("%Y%m%d"))
    # file_path = confirm_dir(file_path,datetime.now().strftime("%Y%m%d%H%M%S"))
    vid_path = confirm_dir(file_path,'raw_video')
    file_name = os.path.join(vid_path,datetime.now().strftime("%Y%m%d%H%M%S")+'.avi')
    file_path = confirm_dir(file_path,'raw_video')
    fourcc = 'mp4v'  # output video codec
    vid_writer = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*fourcc), dataset.fps, (2560, 960))
    with open(os.path.join(file_path,datetime.now().strftime("%Y%m%d%H%M%S")+'ts.txt'),'w') as f:
        for _,img_left,img_right,_,TimeStamp,_ in dataset:
            with timeblock('RPOCESS'):
                frame = cv2.hconcat([img_left,img_right])
                xyxy = [0,0,1,1]
                box_label = str(dataset.count)+'('+str(dataset.frame)+')'+TimeStamp[0]+'.'+TimeStamp[1]
                plot_one_box(xyxy, frame, label=box_label, color=[137,205,36], line_thickness=5) 
                cv2.imshow('test',frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                vid_writer.write(frame)
                line = str(dataset.count)+'('+str(dataset.frame)+')'+':'+str(TimeStamp)+'\n'
                f.write(line)


#%% input port
if __name__ == '__main__':
    tt0=time.time()
    # parameter input with model start up
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="The data source for model input", type=str, default='./data/test08')
    parser.add_argument("--img_size", help="The data size for model input", nargs='+', type=int, default=[416,416])
    parser.add_argument("--tcp_port", help="tcp port", type=int, default=9191)
    parser.add_argument("--tcp_ip", help="tcp ip", type=str, default='192.168.3.181')
    parser.add_argument("--out_range", help="The data size for model input", nargs='+', type=float, default=[0.5,1])
    parser.add_argument("--sm_lambda", help="Stereo matching post filter parameter lambda", type=float, default=8000)
    parser.add_argument("--sm_sigma", help="Stereo matching post filter parameter sigmacolor", type=float, default=1.0)
    parser.add_argument("--sm_UniRa", help="Stereo matching post filter parameter UniquenessRatio", type=int, default=40)
    parser.add_argument("--score", help="inference score threshold", type=float, default=0)
    parser.add_argument("--fps", help="The webcam frequency", type=int, default=1)
    parser.add_argument("--cam_type", help="0: OV9714, 1: AR0135 1280X720; 2: AR0135 1280X960; 3:AR0135 416X416; 4:AR0135 640X640; 5:AR0135 640X480", type=int, default=5)
    parser.add_argument("--ratio", help="ratio for distance calculate", type=float, default=0.05)
    parser.add_argument("--device", help="device on which model runs", type=str,default='pc')
    parser.add_argument("--UMat", help="Use opencv with openCL",action="store_true")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--webcam", help="connect to real camera", action="store_true")
    parser.add_argument("--BM", help="switch to BM alogrithm for depth inference", action="store_true")
    parser.add_argument("--debug", help="save data source for replay", action="store_true")
    parser.add_argument("--visual", help="result visualization", action="store_true")
    parser.add_argument("--save_result", help="inference result save", action="store_true")
    parser.add_argument("--save_path",help="path for result saving",type=str,default="runs/detect/test")                    
    args = parser.parse_args()
    # %% 创建局部函数，加快代码运行
    main()