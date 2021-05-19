#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: main function for YOLOv5 + SGBM model running on RK3399pro platform 
@Date     :2021/04/23 19:56:56
@Author      :Noah
@version      :version : 21042603

'''
import os,logging,sys,argparse,time,socket,math,queue

import cv2
from utils.Stereo_Application import SGBM,BM,disparity_centre
from utils.dataset import DATASET_NAMES,loadfiles,loadcam
from utils.stereoconfig import stereoCamera
from utils.general import scale_coords,confirm_dir,timethis,timeblock,socket_client,calib_type

# %% initial environment
def platform_init(bm_model=False, imgsz=640, device='pc', tcp_address=('192.168.3.181',9191)):
    """
    @description  : initialize AI, SGBM, socket connect and stereo camera calibration parameter
    ---------
    @param  :
        bm_model: type bool. when True, stereo matching use BM model, otherwise, stereo matching use SGBM model
        imgsz: type int, AI model used for image resize
        device: type string, reserved
        tcp_address: type tuple (address,port num),tcp connection address
    -------
    @Returns  : 
        AI: type class RKNNDetector, AI model handler
        SM: type class BM or SGBM, stereo matching model handler
        config: class stereoCamera, stereo camera calibration parameter
        soc_client: class socket_client, tcp connection handler
    -------
    """


# %% load data as pipeline
def LoadData(source='', webcam=False, cam_freq=5, imgsz=(640,640), save_path='',debug=False):
    """
    @description  : load data source ,and prepare for iteration
    ---------
    @param  : 
        source: type string, data folder/file name, or webcam ID
        webcam: type bool, if true, the data source is a real-time webcam, otherwise, a image/images or a video/videos
        cam_freq: type int, only effective when webcam is true, set the frame get from the real-time webcam per second 
        imgsz: type tuple (int,int), reserved
        save_path: type string, the save_path directory for real-time webcam images
    -------
    @Returns  :
        dataset: type class loadcam or loadfiles, a iterator for images stream source
    -------
    """
    
    if webcam:
        dataset = loadcam(source, cam_freq, imgsz, save_path, debug, calib_type.AR0135_416_416)
    else:
        dataset = loadfiles(source, imgsz, save_path)
    return dataset

# %% obejct detection and matching
def object_matching(dataset):
    """
    @description  : a combination of data iteration, object predict ,stereo matching and data transmission process 
    ---------
    @param  :
        ai_model: type class RKNNDetector, AI model handler
        sm_model: type BM or SGBM model, stereo matching model handler
        camera_config: type class stereoCamera, stereo camera calibration parameters
        dataset: type class loadfiles or loadcam, data iterator
        ratio: type float, one of the control parameters of disparity to depth process
        imgsz: type tuple (int,int), used by Image_Rectification function and socket function to resize images for SGBM to the same size as image for object detection
        fps: type int, reserved
        debug: type bool, if true, add post-process image to the packet for socket transimission and save original image for replay on PC
        UMat: type bool, if true, use GPU accelarating SGBM and image rectification process
        soc_client: tpye class socket_client, the tcp transfer handler
    -------
    @Returns  : None
    -------
    """
    
    for _,img_left,img_right,_,TimeStamp,_ in dataset:
        cv2.imshow('image',img_left)
        if cv2.waitKey(1) == ord('q'):
            break


#%% main
def main():
    """
    @description  : used to convert most gloable parameters to partial parameters to accelerate
    ---------
    @param  :None
    -------
    @Returns  :None
    -------
    """
    
    
    print(args)
    source, device, bm_model, imgsz, webcam, cam_freq, ratio, debug, visual, UMat, tcp_port, tcp_ip, save_path= \
        args.source, args.device, args.BM, args.img_size, args.webcam, args.cam_freq, args.ratio, args.debug, args.visual,\
        args.UMat, args.tcp_port, args.tcp_ip, args.save_path    
    if args.verbose:
        logging.basicConfig(filename=os.path.join(save_path,'log.txt'),
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    try:
        len(imgsz)
    except:
        imgsz = (imgsz,imgsz)

    # dataset set up
    dataset = LoadData(source, webcam, cam_freq, imgsz, save_path, debug)

    # dataset iteration and model runs
    object_matching(dataset)
    tt1=time.time()
    print('All Done using (%.2fs)'%(tt1-tt0))

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
    parser.add_argument("--score", help="inference score threshold", type=float, default=0)
    parser.add_argument("--cam_freq", help="The webcam frequency", type=int, default=1)
    parser.add_argument("--cam_type", help="0: OV9714, 1: AR0135 1280X720; 2: AR0135 1280X960", type=int, default=1)
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
