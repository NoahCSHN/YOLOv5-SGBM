#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: main function for YOLOv5 + SGBM model running on RK3399pro platform 
@Date     :2021/04/23 19:56:56
@Author      :Noah
@version      :version : 21052005

'''
import os,logging,sys,argparse,time,math,queue
from datetime import date,datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from threading import Thread
from utils.rknn_detect_yolov5 import  RKNNDetector
from utils.img_preprocess import Image_Rectification
from utils.Stereo_Application import Stereo_Matching,disparity_centre,reproject_3dcloud
from utils.dataset import DATASET_NAMES,loadfiles,loadcam
from utils.stereoconfig import stereoCamera
from utils.general import confirm_dir,timethis,timeblock,socket_client,calib_type,camera_mode,matching_points_gen,plot_one_box

# %% initial environment
def platform_init(imgsz=640, tcp_address=('192.168.3.181',9191), save_path=''):
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
    #init stereo matching model
    cam_mode = camera_mode(args.cam_type)
    soc_client=socket_client(address=tcp_address)
    if Stereo_Matching.count != 0:
        del SM
    SM = Stereo_Matching(cam_mode.mode, args.BM, args.filter,\
                         args.sm_lambda, args.sm_sigma, args.sm_UniRa,\
                         args.sm_numdi, args.sm_mindi, args.sm_block, args.sm_tt,\
                         args.sm_pfc, args.sm_pfs, args.sm_pft,\
                         args.sm_sws, args.sm_sr, args.sm_d12md, save_path)
    #init AI model
    MASKS = [[0,1,2],[3,4,5],[6,7,8]]
    ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    CLASSES = DATASET_NAMES.coco_split_names
    if RKNNDetector.count != 0:
        del AI
    AI = RKNNDetector(model='weights/best_416_coco_split_50.rknn',wh=imgsz,masks=MASKS,anchors=ANCHORS,names=CLASSES)

    return AI,SM,soc_client,cam_mode

# %% load data as pipeline
def LoadData(source='', webcam=False, cam_freq=5, imgsz=(640,640), save_path='',debug=False, cam_mode =5):
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
        dataset = loadcam(source, cam_freq, imgsz, save_path, debug, cam_mode.mode.value)
        config = stereoCamera(mode=cam_mode.mode.value,height=cam_mode.size[1],width=cam_mode.size[0])
    else:
        dataset = loadfiles(source, imgsz, save_path)      
        config = stereoCamera(mode=cam_mode.mode.value,height=cam_mode.size[1],width=cam_mode.size[0])
    return dataset,config

# %% 
def object_matching(ai_model,sm_model,camera_config,dataset,ratio,imgsz,fps,debug,UMat,soc_client,visual,cam_mode,filter, save_path):
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
        visual: tpye bool, if true send the result image through TCP with coordinates, otherwise, only coordinates
        cam_mode: tpye int, the camera configuration number
        filter: tpye bool, if true, the disparity map will be a dense matrix smoothing by a post WLS filter, otherwise, disparity map is a sparse matrix
        save_path: str, save path for result
    -------
    @Returns  : None
    -------
    """
    
    disarity_queue = queue.Queue(maxsize=1)
    pred_queue = queue.Queue(maxsize=1)
    fx = camera_config.cam_matrix_left[0,0]
    fy = camera_config.cam_matrix_left[1,1]
    v = camera_config.cam_matrix_left[0,2]
    u = camera_config.cam_matrix_left[1,2]
    t1 = time.time()
    real_time = time.asctime()
    for _,img_left,img_right,_,TimeStamp,_ in dataset:
        t0 = time.time()
        distance = []
        distance.append(TimeStamp)
        if dataset.mode == 'image' or dataset.mode == 'webcam':
            frame = str(dataset.count)
        else:
            frame = str(dataset.count)+'-'+str(dataset.frame)
        img_raw, img_ai, img_left, img_right, gain, padding=Image_Rectification(camera_config, img_left, img_right, imgsz=imgsz, debug=debug, UMat=UMat, cam_mode=cam_mode)
        # with timeblock('process image:'):
        sm_t = Thread(target=sm_model.run,args=(img_left,img_right,camera_config.Q,disarity_queue,UMat,filter))
        ai_t = Thread(target=ai_model.predict,args=(img_raw, img_ai, gain, padding, pred_queue))
        sm_t.start()
        ai_t.start()
        ai_t.join()
        sm_t.join()
        # %% 
        """ 
        将一张图片的预测框逐条分开，并且还原到原始图像尺寸
        """
        disparity,color_3d = disarity_queue.get()
        # color_depth = color_3d[:,:,2]
        # color_xy = color_3d[:,:,:2]
        preds = pred_queue.get()
        labels = []
        coords = []
        scores = []
        raw_coords = []
        if preds[0] != []:
            labels = preds[0]
            scores = preds[1]
            coords = preds[2]
            raw_coords = preds[3]
        index = 0
        for label,score,box,raw_box in zip(labels,scores,coords,raw_coords):
            if score >= args.score:
                pred = []
                temp_dis = disparity_centre(raw_box, ratio, disparity, color_3d[:,:,2], camera_config.focal_length, camera_config.baseline, camera_config.pixel_size, args.sm_mindi)
                if (temp_dis >= args.out_range[0]*1000) & (temp_dis <= args.out_range[1]*1000):
                    distance.append([label,\
                        float((raw_box[0]-v)*temp_dis/fx),\
                        float((raw_box[3]-u)*temp_dis/fy),\
                        float((raw_box[2]-v)*temp_dis/fx),\
                        float((raw_box[3]-u)*temp_dis/fy),\
                        float(temp_dis)]) # two bottom corners and distance to focal point
                    # distance.append([label,\
                    #     float(color_3d[raw_box[3]-1,raw_box[0]-1,0]),\
                    #     float(color_3d[raw_box[3]-1,raw_box[0]-1,1]),\
                    #     float(color_3d[raw_box[3]-1,raw_box[2]-1,0]),\
                    #     float(color_3d[raw_box[3]-1,raw_box[2]-1,1]),\
                    #     float(temp_dis)]) # two bottom corners and distance to focal point

                # %%
                """ 
                将最终深度结果画到图像里
                """
                xyxy = [raw_box[0],raw_box[1],raw_box[2],raw_box[3]]
                box_label = str(round(temp_dis,2)) #cp3.5
                plot_one_box(xyxy, img_ai, label=box_label, color=DATASET_NAMES.name_color[DATASET_NAMES.coco_split_names.index(label)], line_thickness=1)             
                index += 1
        xyxy = [0,padding[0],1,padding[0]+1]
        box_label = str(TimeStamp[0]+'.'+TimeStamp[1])
        plot_one_box(xyxy, img_ai, label=box_label, color=[137,205,36], line_thickness=1) 
        # %%% send result
        soc_client.send(img_ai, disparity, padding, distance, frame, imgsz, 0.5, visual)
        # %%
        """ 
        save result to local 
        """
        # with timeblock('write file'):
        if args.save_result:
            txt_path = confirm_dir(save_path,'txt')
            # %%
            """
            if dataset.mode == 'image':
                file_path = confirm_dir(save_path,'images')
                save_path = os.path.join(file_path,str(dataset.count)+'.bmp')
                cv2.imwrite(save_path, img_ai)
            elif dataset.mode == 'video' or dataset.mode == 'webcam':
                file_path = os.path.join(save_path,dataset.mode)
                dataset.get_vid_dir(file_path)
                dataset.writer.write(img_ai)
            # %%
            with open(os.path.join(txt_path,'result.txt'),'w') as f:
                f.write('-----------------'+real_time+str(frame)+'-----------------\n')
                for pred in distance[1:]:
                    line = real_time+': '+str(pred[0])+','+str(pred[1])+','+str(pred[2])+','+str(pred[3])+','+str(pred[4])+','+str(pred[5])+'\n'
                    f.write(line)
            """
            """ 
            save boxes
            """
            # %%
            with open(os.path.join(txt_path,str(dataset.count)+'.txt'),'a+') as f:
                for raw_box in raw_coords:
                    line = '['+str(raw_box[0])+','+str(raw_box[1])+']'+','+'['+str(raw_box[2])+','+str(raw_box[3])+']'+'\n'
                    f.write(line)
            # %%
            """ 
            save timestamp
            """
            with open(os.path.join(txt_path,'time_stamp.txt'),'+a') as f:
                line = '$TIMESTAMP,'+str(dataset.count)+'('+str(dataset.frame)+')'+':'+str(TimeStamp)+'-'+str(time.time()-t0)+'\n'
                f.write(line)
            # %%
        if dataset.mode == 'webcam':
            print('frame: %s(%s) Done. (%.3fs);Process: use (%.3fs)'%(frame,dataset.frame,(time.time()-t1),time.time()-t0),end='\r') #cp3.5
        else:
            print('frame: %s Done. (%.3fs);Process: use (%.3fs)'%(frame,(time.time()-t1),time.time()-t0),end='\r') #cp3.5
        t1=time.time()
    

#%% main
def main():
    """
    @description  : used to convert most gloable parameters to partial parameters to accelerate
    ---------
    @param  : None
    -------
    @Returns  : None
    -------
    """
    print(args)
    source, device, bm_model, sm_lambda, sm_sigma, sm_unira, sm_numdisparity, sm_mindisparity, sm_block, sm_TextureThreshold, sm_filter,\
    imgsz, webcam, fps, ratio, debug, visual, UMat, tcp_port, tcp_ip= \
    args.source, args.device, args.BM, args.sm_lambda, args.sm_sigma, args.sm_UniRa,\
    args.sm_numdi, args.sm_mindi, args.sm_block, args.sm_tt,\
    args.filter, args.img_size, args.webcam, args.fps, args.ratio, args.debug, args.visual,\
    args.UMat, args.tcp_port, args.tcp_ip
    save_path = confirm_dir(args.save_path,datetime.now().strftime("%Y%m%d"))
    save_path = confirm_dir(save_path,datetime.now().strftime("%Y%m%d%H%M%S"))
    
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

    # platform init
    AI, SM, soc_client,cam_mode= platform_init(imgsz, (tcp_ip,tcp_port), save_path)

    # dataset set up
    dataset, camera_config = LoadData(source, webcam, fps, imgsz, save_path, debug,cam_mode)

    # dataset iteration and model runs
    object_matching(AI, SM, camera_config, dataset, ratio, imgsz, fps, debug, UMat, soc_client, visual, cam_mode.mode.value, sm_filter, save_path)
    tt1=time.time()
    print('All Done using (%.2fs)'%(tt1-tt0))

#%% input port
if __name__ == '__main__':
    tt0=time.time()
    # parameter input with model start up
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="The data source for model input", type=str, default='0')
    parser.add_argument("--img_size", help="The data size for model input", nargs='+', type=int, default=[416,416])
    parser.add_argument("--tcp_port", help="tcp port", type=int, default=9191)
    parser.add_argument("--tcp_ip", help="tcp ip", type=str, default='192.168.3.181')
    parser.add_argument("--out_range", help="The data size for model input", nargs='+', type=float, default=[0.3,1])
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
    parser.add_argument("--fps", help="The webcam frequency", type=int, default=4)
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
    parser.add_argument("--save_path",help="path for result saving",type=str,default="runs/detect/test/")                    
    args = parser.parse_args()
    # %% 创建局部函数，加快代码运行
    main()

# %%
