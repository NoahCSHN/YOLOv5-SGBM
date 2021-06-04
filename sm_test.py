#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : Stereo Matching test
@Date     :2021/05/28 09:47:18
@Author      :Yyc
@version      :1.0
'''

import os,logging,sys,argparse,time,math,queue

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from threading import Thread
from utils.img_preprocess import Image_Rectification
from utils.Stereo_Application import Stereo_Matching,disparity_centre
from utils.dataset import DATASET_NAMES,loadfiles,loadcam
from utils.stereoconfig import stereoCamera
from utils.general import confirm_dir,timethis,timeblock,socket_client,calib_type,camera_mode

def sm_run():
    #init stereo matching model
    cam_mode = camera_mode(args.cam_type)
    if Stereo_Matching.count != 0:
        del sm_model
    sm_model = Stereo_Matching(cam_mode.mode, args.BM, args.sm_lambda, args.sm_sigma, args.sm_UniRa, args.sm_numdi)

    #data source configuration
    if args.webcam:
        dataset = loadcam(args.source, args.fps, args.img_size, args.save_path, args.debug, cam_mode.mode.value)
        camera_config = stereoCamera(mode=cam_mode.mode.value,height=cam_mode.size[1],width=cam_mode.size[0])
    else:
        dataset = loadfiles(args.source, args.img_size, args.save_path)      
        camera_config = stereoCamera(mode=cam_mode.mode.value,height=cam_mode.size[1],width=cam_mode.size[0])

    #image calibration
    disparity_queue = queue.Queue(maxsize=1)
    for _,img_left,img_right,_,TimeStamp,_ in dataset:
        with timeblock('process'):
            if dataset.mode == 'image' or dataset.mode == 'webcam':
                frame = str(dataset.count)
            else:
                frame = str(dataset.count)+'-'+str(dataset.frame)                    
            img_raw, img_ai, img_left, img_right, gain, padding=Image_Rectification(camera_config, img_left, img_right, imgsz=args.img_size, debug=True, UMat=args.UMat, cam_mode=cam_mode.mode.value)
            sm_model.run(img_left,img_right,disparity_queue,args.UMat)
            disparity = disparity_queue.get()
            if cam_mode.mode == calib_type.AR0135_416_416 or cam_mode.mode == calib_type.AR0135_640_640:
                img_left = np.ravel(img_left)
                img_left = img_left[padding[0]*args.img_size[0]*3:(padding[0]+312)*args.img_size[0]*3]     
                img_left = np.reshape(img_left,(312,416,3))          
                disparity = np.ravel(disparity)
                disparity = disparity[padding[0]*args.img_size[0]:(padding[0]+312)*args.img_size[0]]
                disparity = np.reshape(disparity,(312,416))    
            minVal = np.amin(disparity)
            maxVal = np.amax(disparity)        
            disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=255.0/(maxVal-minVal),beta=-minVal*255.0/(maxVal-minVal)), cv2.COLORMAP_JET)
            # disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=255.0/(maxVal-minVal)), cv2.COLORMAP_JET)
            # disparity_color = cv2.applyColorMap(cv2.normalize(disparity), cv2.COLORMAP_JET)
            path_name = confirm_dir(args.save_path,'BM_'+str(args.BM)+'_'+str(args.sm_lambda)+'_'+str(args.sm_sigma)+'_'+str(args.sm_UniRa)+'_'+str(args.sm_numdi))
            file_name = os.path.join(path_name,dataset.file_name)
            merge = cv2.hconcat([disparity_color,img_ai])
            cv2.imshow('Disparity',merge)
            cv2.imwrite(file_name,merge)
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    tt0=time.time()
    # parameter input with model start up
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="The data source for model input", type=str, default='0')
    parser.add_argument("--img_size", help="The data size for model input", nargs='+', type=int, default=[416,416])
    parser.add_argument("--tcp_port", help="tcp port", type=int, default=9191)
    parser.add_argument("--tcp_ip", help="tcp ip", type=str, default='192.168.3.181')
    parser.add_argument("--out_range", help="The data size for model input", nargs='+', type=float, default=[0.5,1])
    parser.add_argument("--sm_lambda", help="Stereo matching post filter parameter lambda", type=float, default=8000)
    parser.add_argument("--sm_sigma", help="Stereo matching post filter parameter sigmacolor", type=float, default=2.0)
    parser.add_argument("--sm_UniRa", help="Stereo matching post filter parameter UniquenessRatio", type=int, default=40)
    parser.add_argument("--sm_numdi", help="Stereo matching max number disparity", type=int, default=48)
    parser.add_argument("--score", help="inference score threshold", type=float, default=0)
    parser.add_argument("--fps", help="The webcam frequency", type=int, default=1)
    parser.add_argument("--cam_type", help="0: OV9714, 1: AR0135 1280X720; 2: AR0135 1280X960; 3:AR0135 416X416; 4:AR0135 640X640; 5:AR0135 640X480; 6:MIDDLEBURY 416X360", type=int, default=5)
    parser.add_argument("--ratio", help="ratio for distance calculate", type=float, default=0.05)
    parser.add_argument("--device", help="device on which model runs", type=str,default='pc')
    parser.add_argument("--UMat", help="Use opencv with openCL",action="store_true")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--webcam", help="connect to real camera", action="store_true")
    parser.add_argument("--BM", help="switch to BM alogrithm for depth inference", action="store_true")
    parser.add_argument("--debug", help="save data source for replay", action="store_true")
    parser.add_argument("--visual", help="result visualization", action="store_true")
    parser.add_argument("--save_result", help="inference result save", action="store_true")
    parser.add_argument("--save_path",help="path for result saving",type=str,default="runs/detect/test/middlebury")                    
    args = parser.parse_args()
    # %% 创建局部函数，加快代码运行
    sm_run()