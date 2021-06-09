#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : Stereo Matching test
@Date     :2021/05/28 09:47:18
@Author      :Yyc
@version      :1.0
'''

import os,logging,sys,argparse,time,math,queue
from datetime import date

today = date.today()

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from threading import Thread
from utils.img_preprocess import Image_Rectification
from utils.Stereo_Application import Stereo_Matching,reproject_3dcloud
from utils.dataset import DATASET_NAMES,loadfiles,loadcam
from utils.stereoconfig import stereoCamera
from utils.general import confirm_dir,timethis,timeblock,socket_client,calib_type,camera_mode,matching_points_gen

def sm_run():
    #init stereo matching model
    cam_mode = camera_mode(args.cam_type)
    path_name = confirm_dir(args.save_path,today.strftime("%Y%m%d"))
    if args.BM:
        path_name = confirm_dir(path_name,'stereoBM'+\
                                            '_filter_'+str(args.filter)+\
                                            '_'+str(args.sm_lambda)+\
                                            '_'+str(args.sm_sigma)+\
                                            '_'+str(args.sm_UniRa)+\
                                            '_'+str(args.sm_numdi)+\
                                            '_'+str(args.sm_mindi)+\
                                            '_'+str(args.sm_block)+\
                                            '_'+str(args.sm_tt)+\
                                            '_'+str(args.sm_pfc)+\
                                            '_'+str(args.sm_pfs)+\
                                            '_'+str(args.sm_pft)+\
                                            '_'+str(args.sm_sws)+\
                                            '_'+str(args.sm_sr)+\
                                            '_'+str(args.sm_d12md)+\
                                            '_'+str(args.cam_type))
    else:
        path_name = confirm_dir(path_name,'stereoSGBM'+\
                                            '_filter_'+str(args.filter)+\
                                            '_'+str(args.sm_lambda)+\
                                            '_'+str(args.sm_sigma)+\
                                            '_'+str(args.sm_UniRa)+\
                                            '_'+str(args.sm_numdi)+\
                                            '_'+str(args.sm_mindi)+\
                                            '_'+str(args.sm_block)+\
                                            '_'+str(args.sm_tt)+\
                                            '_'+str(args.sm_pfc)+\
                                            '_'+str(args.sm_sws)+\
                                            '_'+str(args.sm_sr)+\
                                            '_'+str(args.sm_d12md)+\
                                            '_'+str(args.cam_type))
    if Stereo_Matching.count != 0:
        del sm_model
    sm_model = Stereo_Matching(cam_mode.mode, args.BM, args.filter,\
                               args.sm_lambda, args.sm_sigma, args.sm_UniRa,\
                               args.sm_numdi, args.sm_mindi, args.sm_block, args.sm_tt,\
                               args.sm_pfc, args.sm_pfs, args.sm_pft,\
                               args.sm_sws, args.sm_sr, args.sm_d12md, path_name)

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
        # with timeblock('process'):
        if dataset.mode == 'image' or dataset.mode == 'webcam':
            frame = str(dataset.count)
        else:
            frame = str(dataset.count)+'-'+str(dataset.frame)                    
        img_raw, img_ai, img_left, img_right, gain, padding=Image_Rectification(camera_config, img_left, img_right, imgsz=args.img_size, debug=True, UMat=args.UMat, cam_mode=cam_mode.mode.value)
        sm_model.run(img_left,img_right,camera_config.Q,disparity_queue,args.UMat,args.filter)
        disparity,color_3d = disparity_queue.get()
        # print('disparity max: %.2f;min: %.2f'%(np.amax(disparity),np.amin(disparity)),end='\r')
        points = []
        try:
            with open(os.path.join('runs/detect/test/txt',str(dataset.count)+'.txt'),'r') as f:
                files = f.readlines()
                for point in files:
                    points.append([int(point.split(',')[1].split(']')[0]),int(point.split(',')[0][1:])])  
        except Exception as e:
            print(e,end='\r')
        stereo_merge = matching_points_gen(disparity,img_left,img_right,points,[0,0])        
        if padding != 0:    
            stereo_merge = np.ravel(stereo_merge)
            stereo_merge = stereo_merge[padding[0]*args.img_size[0]*2:(-(padding[0])*args.img_size[0]*2)]
            stereo_merge = np.reshape(stereo_merge,(-1,832))
            img_ai = np.ravel(img_ai)
            img_ai = img_ai[padding[0]*args.img_size[0]*3:(-(padding[0])*args.img_size[0]*3)]
            img_ai = np.reshape(img_ai,(-1,416,3))          
            disparity = np.ravel(disparity)
            disparity = disparity[padding[0]*args.img_size[0]:(-(padding[0])*args.img_size[0])]
            disparity = np.reshape(disparity,(-1,416))    
            # print('disparity max: %.2f;min: %.2f'%(np.amax(disparity),np.amin(disparity)),end='\r')
            minVal = np.amin(disparity)
            maxVal = np.amax(disparity)
            color_3d = np.ravel(color_3d[:,:,2])
            color_3d = np.divide(color_3d,1000)
            color_3d = color_3d[padding[0]*args.img_size[0]:(-(padding[0])*args.img_size[0])]
            color_3d = np.reshape(color_3d,(-1,416))
            # print('distance max: %.2f;min: %.2f'%(np.amax(color_3d),np.amin(color_3d)),end='\r')
            # minVal = np.amin(color_3d)
            # maxVal = np.amax(color_3d)
        # reproject_3dcloud(img_ai,disparity,camera_config.focal_length,camera_config.baseline)
        disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=255.0/(maxVal-minVal),beta=-minVal*255.0/(maxVal-minVal)), cv2.COLORMAP_JET)
        color_merge = cv2.hconcat([disparity_color,img_ai])
        cv2.imshow('color',color_merge)
        cv2.imshow('object matching',stereo_merge)
        # file_name = os.path.join(path_name,'depth_'+dataset.file_name)
        # cv2.imwrite(file_name,color_merge)
        file_name = os.path.join(path_name,'matching_'+dataset.file_name)
        cv2.imwrite(file_name,stereo_merge)
        if cv2.waitKey(1) == ord('q'):
            break
    time.sleep(2)


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
    parser.add_argument("--sm_UniRa", help="Stereo matching post filter parameter UniquenessRatio", type=int, default=5)
    parser.add_argument("--sm_numdi", help="Stereo matching max number disparity", type=int, default=64)
    parser.add_argument("--sm_mindi", help="Stereo matching min number disparity", type=int, default=-5)
    parser.add_argument("--sm_block", help="Stereo matching blocksize", type=int, default=9)
    parser.add_argument("--sm_tt", help="Stereo matching blocksize", type=int, default=5)        
    parser.add_argument("--sm_pfc", help="Stereo matching PreFilterCap", type=int, default=63)    
    parser.add_argument("--sm_pfs", help="Stereo matching PreFilterSize", type=int, default=9)    
    parser.add_argument("--sm_pft", help="Stereo matching PreFilterType", type=int, default=1)    
    parser.add_argument("--sm_sws", help="Stereo matching SpeckleWindowSize", type=int, default=50)  
    parser.add_argument("--sm_sr", help="Stereo matching SpeckleRange", type=int, default=2)    
    parser.add_argument("--sm_d12md", help="Stereo matching Disp12MaxDiff", type=int, default=1)    
    parser.add_argument("--score", help="inference score threshold", type=float, default=0)
    parser.add_argument("--fps", help="The webcam frequency", type=int, default=1)
    parser.add_argument("--cam_type", help="0: OV9714, 1: AR0135 1280X720; 2: AR0135 1280X960; 3:AR0135 416X416; 4:AR0135 640X640; 5:AR0135 640X480; 6:MIDDLEBURY 416X360", type=int, default=5)
    parser.add_argument("--ratio", help="ratio for distance calculate", type=float, default=0.05)
    parser.add_argument("--device", help="device on which model runs", type=str,default='pc')
    parser.add_argument("--UMat", help="Use opencv with openCL",action="store_true")
    parser.add_argument("--filter", help="Enable post WLS filter",action="store_true")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--webcam", help="connect to real camera", action="store_true")
    parser.add_argument("--BM", help="switch to BM alogrithm for depth inference", action="store_true")
    parser.add_argument("--debug", help="save data source for replay", action="store_true")
    parser.add_argument("--visual", help="result visualization", action="store_true")
    parser.add_argument("--save_result", help="inference result save", action="store_true")
    parser.add_argument("--save_path",help="path for result saving",type=str,default="runs/detect/test/middlebury")                    
    args = parser.parse_args()
    print(args)
    # %% 创建局部函数，加快代码运行
    sm_run()