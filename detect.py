# -*- coding: UTF-8 -*-
'''
Author  : Noah
Date    : 20210408
function: main function for YOLOv5 + SGBM model running on RK3399pro platform
version : 21041902
'''
import os,logging,sys,argparse,time

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from utils.rknn_detect_yolov5 import  RKNNDetector,plot_one_box
from utils.img_preprocess import Image_Rectification
from utils.Stereo_Application import SGBM,BM,disparity_centre
from utils.dataset import DATASET_NAMES,loadfiles,loadcam
from utils.stereoconfig import stereoCamera
from utils.general import scale_coords,confirm_dir,timethis,timeblock

# %% initial environment
def platform_init(bm_model=False, imgsz=640, device='pc'):
    #init stereo matching model
    if bm_model:
        if BM.count != 0:
            del SM
        SM = BM()
    else:
        if SGBM.count != 0:
            del SM
        SM = SGBM()
    config = stereoCamera()
    #init AI model
    MASKS = [[0,1,2],[3,4,5],[6,7,8]]
    ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    CLASSES = DATASET_NAMES.voc_names
    if RKNNDetector.count != 0:
        del AI
    AI = RKNNDetector(model='weights/best_416x416.rknn',wh=imgsz,masks=MASKS,anchors=ANCHORS,names=CLASSES)

    return AI,SM,config

# %% load data as pipeline
def LoadData(source='', webcam=False, cam_freq=5, imgsz=(640,640), save_path=''):
    if webcam:
        dataset = loadcam(source, cam_freq, imgsz, save_path)
    else:
        dataset = loadfiles(source, imgsz, save_path)
    return dataset

# %% obejct detection and matching
def object_matching(ai_model,sm_model,camera_config,dataset,ratio,imgsz,debug,UMat):
    fx = camera_config.cam_matrix_left[0,0]
    fy = camera_config.cam_matrix_left[1,1]
    v = camera_config.cam_matrix_left[0,2]
    u = camera_config.cam_matrix_left[1,2]
    t1 = time.time()
    real_time = time.asctime()
    for path,img_left,img_right,img0,vid_cap in dataset:
        # print('interval time (%.2fs)'%(time.time()-t1))
        distance = []
        if dataset.mode == 'image':
            frame = str(dataset.count)
        else:
            frame = str(dataset.count)+'-'+str(dataset.frame)
        img_left, img_right, img_ai, raw_img_shape=Image_Rectification(camera_config, img_left, img_right, imgsz=imgsz,debug=True,UMat=UMat)
        disparity=sm_model.run(img_left,img_right)
        preds, img_shape = ai_model.predict(img_ai)
        # assert len(labels) == len(boxes),'predict labels not matching boxes'
        #%%%% TODO: 将一张图片的预测框逐条分开，并且还原到原始图像尺寸
        labels = []
        coords = []
        raw_coords = []
        if preds[0] != []:
            labels = preds[0]
            coords = preds[1]
            raw_coords = preds[2]
        # coords = np.asarray(preds[1],dtype=np.float64)
        # labels = preds[0]
        # coords = scale_coords(img_shape,coords,img_ai.shape)
        # coords = coords.astype(int).tolist()
        index = 0
        for label,box,raw_box in zip(labels,coords,raw_coords):
            if len(box):
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
		        # logging.debug(f'depth: {temp}') #cp3.6
                logging.debug('depth: %f',float(temp[4])) #cp3.5
                depth.append(temp[4])
                if (temp[4] >= args.out_range[0]*1000) & (temp[4] <= args.out_range[1]*1000):
                    # distance.append([label,int(box[0]),int(box[1]),int(box[2]),int(box[3]),int(depth[0])-camera_config.focal_length]) # raw boxes and depth
                    distance.append([label,float((box[0]*depth[0]-v)/fx),float((box[3]*depth[0]-u)/fy),float((box[2]*depth[0]-v)/fx),float((box[3]*depth[0]-u)/fy),int(depth[0])]) # two bottum corner and distance to image plane

            #%%%% TODO: 将最终深度结果画到图像里
                if debug:
                    xyxy = [box[0],box[1],box[2],box[3]]
                    # label = f'{label}' #cp3.6
                    label = str(label) #cp3.5
                    plot_one_box(xyxy, img_ai, label=label, line_thickness=2)
                    xyxy = [int((box[0]+box[2])/2)-2*int((box[2]-box[0])*ratio),\
                            int((box[1]+box[3])/2)-2*int((box[3]-box[1])*ratio),\
                            int((box[0]+box[2])/2)+2*int((box[2]-box[0])*ratio),\
                            int((box[1]+box[3])/2)+2*int((box[3]-box[1])*ratio)]
                    # label = f'{depth[0]:.2f}' #cp3.6
                    label = str(round(depth[0],2)) #cp3.5
                    plot_one_box(xyxy, img_ai, label=label, line_thickness=1)                
                index += 1
            #%%%% TODO: 保存结果
        if debug:
            # cv2.namedWindow('Result') #cp3.6
            if dataset.mode == 'image':
                # cv2.imshow('Result',img_ai) #cp3.6
                # cv2.waitKey(1000) #cp3.6
                # plt.imshow(cv2.cvtColor(img_ai,cv2.COLOR_BGR2RGB)) #cp3.5
                # plt.title('result') #cp3.5
                # plt.show() #cp3.5
                file_path = confirm_dir(args.save_path,'images')
                save_path = os.path.join(file_path,str(dataset.count)+'.bmp')
                cv2.imwrite(save_path, img_ai)
            elif dataset.mode == 'video' or dataset.mode == 'webcam':
                file_path = os.path.join(args.save_path,dataset.mode)
                dataset.get_vid_dir(file_path)
                dataset.writer.write(img_ai)
                # cv2.imshow('Result',img_ai) #cp3.6
                # plt.imshow(cv2.cvtColor(img_ai,cv2.COLOR_BGR2RGB)) #cp3.5
                # plt.title('result') #cp3.5
                # plt.show() #cp3.5
        # logging.debug(f'result: {distance}') #cp3.6
        logging.debug('result: %s',distance) #cp3.5
        
        # cv2.destroyAllWindows() #cp3.6
        txt_path = confirm_dir(args.save_path,'txt')
        with open(os.path.join(txt_path,'result'),'a+') as f:
            f.write('-----------------'+real_time+str(frame)+'-----------------\n')
            for pred in distance:
                line = real_time+': '+str(pred[0])+','+str(pred[1])+','+str(pred[2])+','+str(pred[3])+','+str(pred[4])+','+str(pred[5])+'\n'
                f.write(line)
                print('--------------------'+line,end='')
        # logging.info(f'frame: {frame} Done. ({time.time() - t0:.3f}s)') #cp3.6
        print('frame: %s Done. (%.3fs)'%(frame,(time.time()-t1))) #cp3.5
        t1=time.time()
        real_time = time.asctime()

#%% main
def main():
    print(args)
    if args.verbose:
        logging.basicConfig(filename=os.path.join(args.save_path,'log.txt'),
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    source, device, bm_model, imgsz, webcam, cam_freq, ratio, debug, UMat = args.source, args.device, args.BM, args.img_size, args.webcam, args.cam_freq, args.ratio, args.debug, args.UMat
    try:
        len(imgsz)
    except:
        imgsz = (imgsz,imgsz)

    # platform init
    AI, SM, camera_config= platform_init(bm_model, imgsz, device)

    # dataset set up
    dataset = LoadData(source, webcam, cam_freq, imgsz, args.save_path)

    # dataset iteration and model runs
    object_matching(AI, SM, camera_config, dataset, ratio, imgsz, debug, UMat)
    tt1=time.time()
    print('All Done using (%.2fs)'%(tt1-tt0))

#%% input port
if __name__ == '__main__':
    tt0=time.time()
    # parameter input with model start up
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="The data source for model input", type=str, default='./data/test08')
    parser.add_argument("--img_size", help="The data size for model input", type=int, default=[416,416])
    parser.add_argument("--out_range", help="The data size for model input", type=float, default=[0.5,1])
    parser.add_argument("--cam_freq", help="The webcam frequency", type=int, default=5)
    parser.add_argument("--ratio", help="ratio for distance calculate", type=float, default=0.05)
    parser.add_argument("--device", help="device on which model runs", type=str,default='pc')
    parser.add_argument("--UMat", help="Use opencv with openCL",action="store_true")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--webcam", help="connect to real camera", action="store_true")
    parser.add_argument("--BM", help="switch to BM alogrithm for depth inference", action="store_true")                    
    parser.add_argument("--debug", help="run under debug mode", action="store_true")
    parser.add_argument("--save_path",help="path for result saving",type=str,default="runs/detect/test")                    
    args = parser.parse_args()
    # %% 创建局部函数，加快代码运行
    main()

    
