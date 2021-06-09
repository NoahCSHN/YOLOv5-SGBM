'''
Author  : Noah
Date    : 20210408
function: common-basic function database
'''
import os,sys,logging,cv2,time,functools,socket
from pathlib import Path
import numpy as np
from contextlib import contextmanager
from enum import Enum

def timethis(func):
    """
    @description  : a timecounter wrapper for a function
    ---------
    @param  :
    -------
    @Returns  : print the runtime and name of the function to the terminal 
    -------
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {:.3f}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper

@contextmanager
def timeblock(label):
    """
    @description  : a timecounter wrapper for a section of code
    ---------
    @param  :
    -------
    @Returns  : print the runtime of a section of code to the terminal 
    -------
    """   
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('{} : {}'.format(label, end - start),end='\r')


def get_new_size(img, scale):
    if type(img) != cv2.UMat:
        return tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    else:
        return tuple(map(int, np.array(img.get().shape[:2][::-1]) * scale))

def get_max_scale(img, max_w, max_h):
    if type(img) == cv2.UMat:
        h, w = img.get().shape[:2]
    else:
        h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    return scale

class AutoScale:
    def __init__(self, img, max_w, max_h):
        self._src_img = img
        self.scale = get_max_scale(img, max_w, max_h)
        self._new_size = get_new_size(img, self.scale)
        self.__new_img = None

    @property
    def size(self):
        return self._new_size

    @property
    def new_img(self):
        if self.__new_img is None:
            self.__new_img = cv2.resize(self._src_img, self._new_size,interpolation=cv2.INTER_AREA)
        return self.__new_img

# @timethis
def letterbox(img, new_wh=(416, 416), color=(114, 114, 114)):
    a = AutoScale(img, *new_wh)
    new_img = a.new_img
    if type(new_img) == cv2.UMat:
        h, w = new_img.get().shape[:2]
    else:
        h, w = new_img.shape[:2]
    padding = [(new_wh[1] - h)-int((new_wh[1] - h)/2), int((new_wh[1] - h)/2), (new_wh[0] - w)-int((new_wh[0] - w)/2), int((new_wh[0] - w)/2)]
    # new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - h, 0, new_wh[0] - w, cv2.BORDER_CONSTANT, value=color)
    new_img = cv2.copyMakeBorder(new_img, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT, value=color)
    # logging.debug(f'image padding: {padding}') #cp3.6
    logging.debug('image padding: %s',padding) #cp3.5
    return new_img, (new_wh[0] / a.scale, new_wh[1] / a.scale), padding

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    @description  :change pixel coordinates in the img1_shape to img0_shape
    ---------
    @param  :
        img1_shape: type tuple(int,int), the image size of the coords
        coords: type numpy.array, the coords for translation
        img0_shape: type tuple(int,int), the expected image size of the coords after translation
        ratio_pad: type tuple(int,(int,int)), tuple(0) is the resize ratio, tuple(1,0) is the x padding, tuple(1,1) is the y padding
    -------
    @Returns  :
        coords: type numpy.array, the coordinates in img1_shape
    -------
    """
    
    
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords = clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = clip_axis(boxes[:,0], img_shape[1])  # x1
    boxes[:, 1] = clip_axis(boxes[:,1], img_shape[0])  # y1
    boxes[:, 2] = clip_axis(boxes[:,2], img_shape[1])  # x2
    boxes[:, 3] = clip_axis(boxes[:,3], img_shape[0])  # y2
    return boxes

def clip_axis(axis,limit):
    ind = np.where(axis>limit)
    axis[ind] = limit
    ind = np.where(axis<0)
    axis[ind] = 0
    return axis

def confirm_dir(root_path,new_path):
    """
    @description  :make sure path=(root_path+new_path) exist
    ---------
    @param  :
        root_path: type string, 
        new_path: type string,
    -------
    @Returns  :
    -------
    """
    
    
    path = os.path.join(root_path,new_path)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

class socket_client():
    """
    @description  : handle tcp event
    ---------
    @function  :
    -------
    """
    
    
    def __init__(self,address=('192.168.3.181',9191)):
        """
        @description  : obtain and save the tcp client address
        ---------
        @param  : address: server address, in default taple format
        -------
        @Returns  : None
        -------
        """
        self.address = address
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(self.address)
            print('connect to: ',address)
        except socket.error as msg:
            self.sock = None
            print(msg)
            sys.exit(1)         
    
    # @timethis
    def send(self,image,disparity,padding,coords,frame,imgsz=(416,416),quality=0.5,debug=False):
        """
        @description  : send a packet to tcp server
        ---------
        @param  : image: the cv2 image mat (imgsz[0],imgsz[1])
        @param  : coords: the coords of the opposite angle of the object rectangle,(n,(x1,y1,z1,x2,y2,z2))
        @param  : frame: frame number of the pipe stream
        @param  : imgsz: the image resolution(height,width), reserved
        @param  : quality: type float ,in the range [0,1], reserved
        @param  : debug: type bool, if true, add a image in imgsz shape to tcp transmission packet
        -------
        @Returns  : None
        -------
        """
        
        answer = []
        if debug:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality*100)]
            ## 首先对图片进行编码，因为socket不支持直接发送图片
            stringData = np.ravel(image)
            stringData = stringData[padding[0]*imgsz[0]*3:(padding[0]+312)*imgsz[0]*3]       
            disparity = np.ravel(disparity)
            disparity = disparity[padding[0]*imgsz[0]:(padding[0]+312)*imgsz[0]]
            minVal = np.amin(disparity)
            maxVal = np.amax(disparity)
            disparity = np.reshape(disparity,(312,416))
            disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=255.0/(maxVal-minVal),beta=-minVal*255.0/(maxVal-minVal)), cv2.COLORMAP_JET)
            colorsz = disparity_color.shape
            ## 首先发送图片编码后的长度
            header='$Image,'+str(len(stringData))+','+str(len(np.ravel(disparity_color)))+','+str(312)+','+str(416)+','+str(colorsz[0])+','+str(colorsz[1])+','+str(coords[0][0])+','+str(coords[0][1])+','+str(frame)
            self.sock.sendall(header.encode('utf-8').ljust(64)) 
            while answer != 'Ready for Image':
                answer = self.sock.recv(32).decode('utf-8')
            self.sock.sendall(stringData)
            while answer != 'Ready for Disparity Image':
                answer = self.sock.recv(64).decode('utf-8')
            self.sock.sendall(np.ravel(disparity_color))
        else:
            header='$Image,'+str('0')+','+str(coords[0][0])+','+str(coords[0][1])
            self.sock.sendall(header.encode('utf-8').ljust(32))
        while answer != 'Ready for Coordinates':
            answer = self.sock.recv(32).decode('utf-8')

        coordData = '$Coord'
        for item in coords[1:]:
            coordData += ','
            coordData += str(item[1])
            coordData += ','
            coordData += str(item[2])
            coordData += ','
            coordData += str(item[5])
            coordData += ','
            coordData += str(item[3])
            coordData += ','
            coordData += str(item[4])
            coordData += ','
            coordData += str(item[5])
        coordData += ',*FC'
        ## 然后发送编码的内容
        self.sock.sendall(coordData.encode('utf-8'))
        while answer != 'Ready for next Frame':
            answer = self.sock.recv(32).decode('utf-8')
    
    def close(self):
        if self.sock:
            print('closing tcp client ...')
            self.sock.close()
    
    def __del__(self):
        self.close()

class calib_type(Enum):
    """
    @description  : camera type for image rectification
    ---------
    @function  : sequence the camera type by number
    -------
    """
    OV9714_1280_720 = 0
    AR0135_1280_720 = 1
    AR0135_1280_960 = 2
    AR0135_416_416  = 3
    AR0135_640_640  = 4
    AR0135_640_480  = 5
    MIDDLEBURY_416  = 6

class camera_mode:
    """
    @description  : camera mode for image rectification
    ---------
    @function  : give each calib type a image size for rectification
    -------
    """
    def __init__(self,mode):
        if mode == 0:
            self.mode=calib_type.OV9714_1280_720
            self.size=(1280,720)
        elif mode == 1:
            self.mode=calib_type.AR0135_1280_720
            self.size=(1280,720)
        elif mode == 2:
            self.mode=calib_type.AR0135_1280_960
            self.size=(1280,960)
        elif mode == 3:
            self.mode=calib_type.AR0135_416_416
            self.size=(416,416)
        elif mode == 4:
            self.mode=calib_type.AR0135_640_640
            self.size=(640,640)
        elif mode == 5:
            self.mode=calib_type.AR0135_640_480
            self.size=(640,480)
        else:
            self.mode=calib_type.MIDDLEBURY_416
            self.size=(1280,960)

def matching_points_gen(disparity,img_left,img_right,left_points=[],padding=[]):
    """
    @description  : get the left image points and draw a line between the original point in the image_left and matching point in the image_right
    ---------
    @param  : disparity: type matrix, the disparity map of the image_left and image_right
    @param  : image_left: type matrix, the raw image from the left camera
    @param  : image_right: type matrix, the raw image from the right camera
    @param  : left_points: type list, the point in the image_left
    -------
    @Returns  : the image with matching points line in it
    -------
    """
    merge = cv2.hconcat([img_left,img_right])
    if left_points == []:
        return merge
    
    # %% 加点
    raw_points = []
    for i in range(int(len(left_points)/2)):
        add_point = [int((left_points[2*i][0]+left_points[2*i+1][0])/2),int((left_points[2*i][1]+left_points[2*i+1][1])/2)]
        raw_points.append(left_points[2*i])
        raw_points.append(add_point)
        raw_points.append(left_points[2*i+1])

    # %% 划线
    for point in raw_points:
        # print(padding[0])
        first_matching_point = [point[1]-1+416-padding[0],point[0]-1]
        first_point = [point[1]-1-padding[0],point[0]-1]
        cv2.line(merge,first_point,first_matching_point,color=(0,255,0),thickness=1,lineType=cv2.LINE_8)    
        sec_matching_point = [int(point[1]-1+416-disparity[point[1]-1,point[0]-1])-padding[0],point[0]-1]
        sec_point = [point[1]-1+416-padding[0],point[0]-1]
        cv2.line(merge,sec_point,sec_matching_point,color=(0,255,0),thickness=2,lineType=cv2.LINE_8)    
    return merge
    