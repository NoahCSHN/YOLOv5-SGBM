#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: main function for YOLOv5 + SGBM model running on RK3399pro platform 
@Date     :2021/04/23 19:56:56
@Author      :Noah
@version      :version : 21042603

'''
import os,logging,sys,argparse,time,socket

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from utils.rknn_detect_yolov5 import  RKNNDetector,plot_one_box
from utils.img_preprocess import Image_Rectification
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
    
    
    #init stereo matching model
    soc_client=socket_client(address=tcp_address)
    if bm_model:
        if BM.count != 0:
            del SM
        SM = BM()
    else:
        if SGBM.count != 0:
            del SM
        SM = SGBM()
    config = stereoCamera(mode=calib_type.AR0135_1280_960,height=960,width=1280)
    #init AI model
    MASKS = [[0,1,2],[3,4,5],[6,7,8]]
    ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    CLASSES = DATASET_NAMES.coco_split_names
    if RKNNDetector.count != 0:
        del AI
    AI = RKNNDetector(model='weights/best_416_coco_split_50.rknn',wh=imgsz,masks=MASKS,anchors=ANCHORS,names=CLASSES)

    return AI,SM,config,soc_client

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
        dataset = loadcam(source, cam_freq, imgsz, save_path, debug, calib_type.AR0135_1280_960)
    else:
        dataset = loadfiles(source, imgsz, save_path)
    return dataset

# %% obejct detection and matching
def object_matching(ai_model,sm_model,camera_config,dataset,ratio,imgsz,fps,debug,UMat,soc_client,visual):
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
        # print('TimeStamp: ',TimeStamp)
        # print('TimeStamp: ',distance)
        if dataset.mode == 'image' or dataset.mode == 'webcam':
            frame = str(dataset.count)
        else:
            frame = str(dataset.count)+'-'+str(dataset.frame)
        img_left, img_right, img_ai, _=Image_Rectification(camera_config, img_left, img_right, imgsz=imgsz,debug=True,UMat=UMat)
        disparity=sm_model.run(img_left,img_right)
        preds, _ = ai_model.predict(img_ai)
        # assert len(labels) == len(boxes),'predict labels not matching boxes'
        #%%%% TODO: 将一张图片的预测框逐条分开，并且还原到原始图像尺寸
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
                cxcy = []
                depth = [] #distance calculate 

            #%%%% TODO: 分9个图像框
                dx,dy=int((raw_box[2]-raw_box[0])*ratio),int((raw_box[3]-raw_box[1])*ratio)
                cx,cy=int((raw_box[0]+raw_box[2])/2),int((raw_box[1]+raw_box[3])/2)
                dw,dh=int((raw_box[2]-raw_box[0])/6),int((raw_box[3]-raw_box[1])/6)
                cxcy=[(cx-2*dw,cy-2*dh),(cx,cy-2*dh),(cx+2*dw,cy-2*dh),\
                    (cx-2*dw,cy),(cx,cy),(cx+2*dw,cy),\
                    (cx-2*dw,cy+2*dh),(cx,cy+2*dh),(cx+2*dw,cy+2*dh)]

            #%%%% TODO: 每个框计算深度均值
                temp = np.zeros((9,),dtype=float)
                k = 0
                for m,n in cxcy:
                    temp[k] = disparity_centre(m, n, dx, dy, disparity, camera_config.focal_length, camera_config.baseline, camera_config.pixel_size)
                    k += 1
            
            #%%%% TODO: 取众多框计算值的中位数
                temp = np.sort(temp)
                temp = temp[temp!=-1.]
                # print('depth: ',temp)
            if len(temp) == 0:
                temp_dis == -1
            elif (len(temp)%2 == 0) & (len(temp)>1):
                temp_dis = (temp[round(len(temp)/2)]+temp[round(len(temp)/2)-1])/2
            else:
                temp_dis = temp[round(len(temp)/2)]
                #if (len(temp)%2 == 0) & (len(temp)>1):
                #    temp_dis = (temp[round(len(temp)/2)]+temp[round(len(temp)/2)-1])/2 
                #else:
                #    temp_dis = temp[round(len(temp)/2)]
                depth.append(temp_dis)
                if (temp_dis >= args.out_range[0]*1000) & (temp_dis <= args.out_range[1]*1000):
                    # distance.append([label,int(box[0]),int(box[1]),int(box[2]),int(box[3]),int(depth[0])-camera_config.focal_length]) # raw boxes and depth
                    distance.append([label,\
                        float((box[0]-v)*depth[0]/fx),\
                        float((box[3]-u)*depth[0]/fy),\
                        float((box[2]-v)*depth[0]/fx),\
                        float((box[3]-u)*depth[0]/fy),\
                        int(depth[0])]) # two bottom corners and distance to focal point

            # %%%% TODO: 将最终深度结果画到图像里
                # if debug:
                # if False:
                xyxy = [box[0],box[1],box[2],box[3]]
                label = str(label)+':'+str(round(score,2)) #cp3.5
                plot_one_box(xyxy, img_ai, label=label, line_thickness=3)
                xyxy = [int((box[0]+box[2])/2)-2*int((box[2]-box[0])*ratio),\
                        int((box[1]+box[3])/2)-2*int((box[3]-box[1])*ratio),\
                        int((box[0]+box[2])/2)+2*int((box[2]-box[0])*ratio),\
                        int((box[1]+box[3])/2)+2*int((box[3]-box[1])*ratio)]
                label = str(round(depth[0],2)) #cp3.5
                plot_one_box(xyxy, img_ai, label=label, line_thickness=3)             
                index += 1
            #%%%% TODO: 保存结果
        # with timeblock('write file'):
        if visual:
            if dataset.mode == 'image':
                file_path = confirm_dir(args.save_path,'images')
                save_path = os.path.join(file_path,str(dataset.count)+'.bmp')
                cv2.imwrite(save_path, img_ai)
            elif dataset.mode == 'video' or dataset.mode == 'webcam':
                file_path = os.path.join(args.save_path,dataset.mode)
                dataset.get_vid_dir(file_path)
                dataset.writer.write(img_ai)
        # logging.debug('result: %s',distance) #cp3.5
        
        # cv2.destroyAllWindows() #cp3.6
        txt_path = confirm_dir(args.save_path,'txt')
        with open(os.path.join(txt_path,'result'),'a+') as f:
            f.write('-----------------'+real_time+str(frame)+'-----------------\n')
            for pred in distance[1:]:
                line = real_time+': '+str(pred[0])+','+str(pred[1])+','+str(pred[2])+','+str(pred[3])+','+str(pred[4])+','+str(pred[5])+'\n'
                f.write(line)
                print(line,end='')
        # logging.info(f'frame: {frame} Done. ({time.time() - t0:.3f}s)') #cp3.6
        if True:
            print('frame: %s Done. (%.3fs)'%(frame,(time.time()-t1)),end='') #cp3.5
            print(' --Process: use (%.3fs)'%(time.time()-t0)) #cp3.5
        t1=time.time()
        real_time = time.asctime()
        soc_client.send(img_ai, distance, fps, imgsz, visual)

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

    # platform init
    AI, SM, camera_config, soc_client= platform_init(bm_model, imgsz, device, (tcp_ip,tcp_port))

    # dataset set up
    dataset = LoadData(source, webcam, cam_freq, imgsz, save_path, debug)

    # dataset iteration and model runs
    object_matching(AI, SM, camera_config, dataset, ratio, imgsz, cam_freq, debug, UMat, soc_client,visual)
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
    parser.add_argument("--save_path",help="path for result saving",type=str,default="runs/detect/test")                    
    args = parser.parse_args()
    # %% 创建局部函数，加快代码运行
    main()
