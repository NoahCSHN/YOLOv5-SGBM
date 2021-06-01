'''
Author  : Noah
Date    : 20210408
function: Load data to input to the model
'''
import os,sys,logging,glob,time,rospy
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import ThreadPool
from threading import Thread
# from utils.general import confirm_dir,timethis,calib_type

import cv2

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

class DATASET_NAMES():
    """
    @description  :pre define the object class name for object detection
    ---------
    @function  : None
    -------
    """
    
    voc_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                    'hair drier', 'toothbrush']
    masks_names = ['mask','nomask']
    voc_split_names = ['bottle','chair','diningtable','person','pottedplant','sofa','tvmonitor']
    coco_split_names = ['person','sports ball','bottle','cup','chair','potted plant','cell phone', 'book']

class loadfiles:
    """
    @description  : load iamge or video file(s) and create a iterator
    ---------
    @function  :
    -------
    """
    
    
    def __init__(self, path='', img_size=640, save_path='/home/bynav/RK3399/AI_SGBM/runs/'):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            try:
                files = sorted(glob.glob(os.path.join(p, '*.*')), key=lambda x: int(os.path.basename(x).split('.')[0]))  # dir
            except:
                files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist'%p) #cp3.5

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.vid_file_path = os.path.join(save_path,'video')
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.writer = None          #debug function
        if any(videos):
            self.new_video(videos[0])  # new video
        assert self.nf > 0, 'No images or videos found in %s. '%p \
                            # 'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}' #cp3.5

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if isinstance(self.writer, cv2.VideoWriter):
                    self.writer.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %d/%d (%d/%d) %s: '%(self.count + 1,self.nf,self.frame,self.nframes,path), end='') #cp3.5

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('========================new image========================')
            print('image %d/%d %s: '%(self.count, self.nf, path), end='\n') #cp3.5

        # Padded resize


        return path, img0, self.cap
    
    def get_vid_dir(self,path):
        self.vid_file_path = path
    
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        # self.file_path = '/home/bynav/AI_SGBM/runs/detect/exp/video'
        if not os.path.isdir(self.vid_file_path):
            os.mkdir(self.vid_file_path)
        save_path = os.path.join(self.vid_file_path, str(path+'.avi'))
        fourcc = 'mp4v'  # output video codec
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), 2, (1280, 960))
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

if __name__ == '__main__':
    dataset = loadfiles(path='/home/bynav/RK3399/AI_SGBM/runs/detect/test/raw_video')
    save_path = '/home/bynav/RK3399/AI_SGBM/runs/detect/test/raw_image'
    for _,img,_ in dataset:
        if dataset.mode == 'image':
            dataset.writer.write(img)
            cv2.imshow('Test',img)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            save_name=os.path.join(save_path,str(dataset.frame)+'.png')
            cv2.imwrite(save_name,img)
            cv2.imshow('Test',img)
            if cv2.waitKey(1) == ord('q'):
                break