'''
Author  : Noah
Date    : 20210408
function: Load data to input to the model
'''
import os,sys,logging,glob,time,queue
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import ThreadPool
from threading import Thread
import threading
import numpy as np
from utils.general import confirm_dir,timethis,calib_type

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
    name_color = [[246, 252, 48],[38, 9, 129],[235, 204, 85],[51, 148, 36],[68, 154, 71],[77, 204, 64],[142, 183, 11],
                  [76, 224, 194],[62, 211, 108],[87, 182, 84],[217, 236, 51],[83, 160, 30],[136, 38, 28],[157, 71, 128],
                  [166, 144, 72],[142, 82, 203],[161, 110, 0],[179, 75, 107],[241, 31, 58],[188, 179, 151],[6, 141, 72],
                  [34, 65, 134],[248, 200, 119],[98, 14, 74],[108, 42, 45],[65, 253, 19],[41, 70, 255],[72, 54, 7],
                  [86, 8, 97],[106, 129, 218],[59, 147, 175],[234, 40, 195],[92, 42, 230],[236, 173, 62],[144, 190, 177],
                  [18, 181, 241],[247, 59, 100],[212, 181, 95],[143, 117, 204],[30, 46, 171],[86, 254, 78],[82, 124, 249],
                  [142, 236, 83],[193, 223, 226],[198, 202, 19],[101, 171, 24],[212, 147, 16],[55, 73, 49],[104, 91, 136],
                  [205, 89, 132],[42, 103, 28],[109, 60, 150],[250, 216, 158],[211, 132, 120],[188, 40, 169],[92, 12, 162],
                  [107, 64, 221],[149, 174, 193],[126, 54, 154],[88, 107, 46],[115, 128, 33],[73, 202, 252],[1, 224, 125],
                  [9, 55, 163],[66, 145, 204],[61, 248, 181],[220, 238, 17],[53, 26, 250],[162, 156, 200],[240, 117, 64],
                  [53, 65, 194],[17, 146, 93],[197, 199, 158],[64, 54, 35],[188, 183, 177],[206, 17, 174],[34, 155, 144],
                  [142, 123, 110],[211, 17, 89],[54, 38, 67]]

class pipeline:
    """
    @description  : a data pipeline shared by multiple threads
    ---------
    @function  : send images, timestamps and valid signal
    -------
    """
    def __init__(self,width=2560,height=960):
        self.timestamp=0.
        self.frame=0
        self.valid = False
        self.lock = threading.Lock()

    def put(self,timestamp,img0,frame):
        with self.lock:
            self.timestamp = timestamp
            self.frame = frame
            self.image = img0
            self.valid = True
    
    def get(self):
        with self.lock:
            timestamp=self.timestamp
            img0=self.image
            frame = self.frame
            valid = self.valid
            self.valid = False
            return timestamp,img0,frame,valid

class loadfiles:
    """
    @description  : load iamge or video file(s) and create a iterator
    ---------
    @function  :
    -------
    """
    
    
    def __init__(self, path='', img_size=640, save_path=''):
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
        self.file_name = 'Orign'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'Supported formats are:\nimages: %s\nvideos: %s'%(img_formats,vid_formats) #cp3.5
                            # 'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}' #cp3.6

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

            self.file_name = path+'_'+self.frame
            self.frame += 1
            # print('video %d/%d (%d/%d) %s: '%(self.count + 1,self.nf,self.frame,self.nframes,path), end='') #cp3.5

        else:
            # Read image
            self.count += 1
            self.mode = 'image'
            img0 = cv2.imread(path)  # BGR
            self.file_name = os.path.split(path)[-1]
            assert img0 is not None, 'Image Not Found ' + path
            # print('========================new image========================')
            # print('image %d/%d %s: '%(self.count, self.nf, path), end='\n') #cp3.5

        # Padded resize
        TimeStamp = str(time.time()).split('.')
        if len(TimeStamp[1])<9:
            for i in range(9-len(TimeStamp[1])):
                TimeStamp[1] += '0'
        h = img0.shape[0]
        w = img0.shape[1]        
        w1 = round(w/2)
        img0_left = img0[:,:w1,:]
        img0_right = img0[:,w1:,:]

        return path, img0_left, img0_right, (h,w1), TimeStamp, self.cap
    
    def get_vid_dir(self,path):
        self.vid_file_path = path
    
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        # self.file_path = '/home/bynav/AI_SGBM/runs/detect/exp/video'
        if not os.path.isdir(self.vid_file_path):
            os.mkdir(self.vid_file_path)
        save_path = os.path.join(self.vid_file_path, str(path.split('/')[-1].split('.')[0])+'.avi')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = 'mp4v'  # output video codec
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

class loadcam:
    """
    @description  : load real-time webcam data and create a iterator
    ---------
    @function  :
    ---------
    """
    
    # @timethis
    def __init__(self, pipe='4', cam_freq=5, img_size=640, save_path='', debug=False, cam_mode=1):
        self.img_size = img_size
        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.debug = debug
        self.pipe = pipe
        self.time = 0
        self.writer = None
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        if cam_mode == calib_type.AR0135_1280_960.value or cam_mode == calib_type.AR0135_416_416.value or cam_mode == calib_type.AR0135_640_640.value:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,960) #AR0135
        elif cam_mode == calib_type.AR0135_640_480.value:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480) #AR0135
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720) #OV9714
        self.pipeline = pipeline(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.cam_freq = cam_freq
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.queue = queue.LifoQueue(maxsize=self.fps)
        bufsize = 1
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, bufsize)  # set buffer size
        print('Camera run under %s and %s fps'%(str(self.size),str(self.fps)))
        self.vid_file_path = confirm_dir(save_path,'webcam')
        self.img_file_path = confirm_dir(save_path,'webimg')
        self.new_video('test.avi')
        self.mode = 'webcam'
        self.count = 0
        self.frame = 0
        self.real_frame = 0
        self.valid = False
        self.start = False
        self.thread = Thread(target=self._update,args=[],daemon=True)
        self.thread.start()
    
    def _update(self):
        while True:
            self.real_frame += 1
            TimeStamp = time.time()-0.044  #cv2.cap.read() average latency is 290ms
            # Read frame
            if self.pipe in [0,1,2,3,4,5]:  # local camera
                ret_val, img0 = self.cap.read()
                # self.valid = True
                # img0 = cv2.flip(img0, 1)  # flip left-right
            else:  # IP camera
                n = 0
                while True:
                    n += 1
                    self.cap.grab()
                    if n % 30 == 0:  # skip frames
                        ret_val, img0 = self.cap.retrieve()
                        if ret_val:
                            break
            assert ret_val, 'Camera Error %d'%self.pipe #cp3.5
            self.pipeline.put(TimeStamp,img0,self.real_frame)
            self.start = True

    def __iter__(self):
        return self

    # @timethis
    def __next__(self):
        runtime = time.time() - self.time
        if runtime < 1/self.cam_freq:
            time.sleep(round(1/self.cam_freq-runtime,3))
        while True:
            if self.start:
                TimeStamp,img0,self.frame,self.valid = self.pipeline.get()
                if self.valid:
                    break
        # print('========================= webcam %d ======================='%self.frame,end='\r') #cp3.5    
        TimeStamp = str(TimeStamp).split('.')
        if len(TimeStamp[1])<9:
            for i in range(9-len(TimeStamp[1])):
                TimeStamp[1] += '0'         
        w = img0.shape[1]
        w1 = int(w/2)
        if self.debug:
            save_file = os.path.join(self.img_file_path,(str(self.frame)+'.bmp'))
            cv2.imwrite(save_file,img0)
        imgl = img0[:,:w1,:]
        imgr = img0[:,w1:,:]
        self.count += 1
        img_path = 'webcam.jpg'
        self.time = time.time()
        return img_path, imgl, imgr, None, TimeStamp, None

    def get_vid_dir(self,path):
        self.vid_file_path = path

    def new_video(self, path):
        if isinstance(self.writer, cv2.VideoWriter):
            self.writer.release()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = 'mp4v'  # output video codec
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_path = os.path.join(self.vid_file_path, path)
        self.writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    def __len__(self):
        return 0
