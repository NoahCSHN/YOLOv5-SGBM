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
            self.__new_img = cv2.resize(self._src_img, self._new_size)
        return self.__new_img

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
        print('{} : {}'.format(label, end - start))


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
    
    def send(self,image,coords,fps,imgsz=(416,416),debug=False):
        """
        @description  : send a packet to tcp server
        ---------
        @param  :
            image: the cv2 image mat (imgsz[0],imgsz[1])
            coords: the coords of the opposite angle of the object rectangle,(n,(x1,y1,z1,x2,y2,z2))
            fps: the fps of the camera
            imgsz: the image resolution(height,width)
            debug: type bool, if true, add a image in imgsz shape to tcp transmission packet        
        -------
        @Returns  : None
        -------
        """
        
        t0=time.time()
        answer = []

        # print('Start image and coordinate transformation')
        if debug:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), fps]
            ## 首先对图片进行编码，因为socket不支持直接发送图片
            # t1=time.time()
            image,_,_ = letterbox(image,imgsz)
            _, imgencode = cv2.imencode('.jpg', cv2.UMat(image))#
            data = np.array(imgencode)
            stringData = data.tostring()
            # print('image encode: (%s)'%(time.time()-t1))
            ## 首先发送图片编码后的长度
            header='$Image,'+str(len(stringData))+','+str(coords[0][0])+','+str(coords[0][1])
            # print(header)
            self.sock.sendall(header.encode('utf-8').ljust(128)) 
            # print('Send Image size done, waiting for answer')
            while answer != 'Ready for Image':
                answer = self.sock.recv(32).decode('utf-8')
            # print('Recv from server: %s'%answer)
            self.sock.sendall(stringData)
            # print('Send Image done, waiting for answer')
        else:
            header='$Image,'+str('0')+','+str(coords[0][0])+','+str(coords[0][1])
            self.sock.sendall(header.encode('utf-8').ljust(32))
        while answer != 'Ready for Coordinates':
            answer = self.sock.recv(32).decode('utf-8')
        # print('Recv from server: %s'%answer)

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
        # print(coordData)
        self.sock.sendall(coordData.encode('utf-8'))
        # print('Send Coordinates, waiting for answer')
        while answer != 'Ready for next Frame':
            answer = self.sock.recv(32).decode('utf-8')
        # print('Recv from server: %s'%answer)
        print('TCP transport use: %0.3f'%(time.time()-t0))
        # self.sock.close()
    
    def close(self):
        if self.sock:
            print('closing tcp client ...')
            self.sock.close()
    
    def __del__(self):
        self.close()

class calib_type(Enum):
    OV9714_1280_720 = 0
    AR0135_1280_720 = 1
    AR0135_1280_960 = 2
