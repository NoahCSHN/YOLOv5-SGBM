import numpy as np
import argparse,logging,time,math
import cv2
from matplotlib import pyplot as plt
from utils.general import calib_type, timethis

class Cursor:
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.ax.figure.canvas.draw()

class BM:
    """
    @description  : 'BM Algorithm class'
    ---------
    @function  :
    -------
    """
    count=0
    def __init__(self,cam_mode):
        t0 = time.time()
        BM.count += 1
        if cam_mode == calib_type.AR0135_640_480:
            self.stereo = cv2.StereoBM_create(64, 9)
        else:
            self.stereo = cv2.StereoBM_create(48, 9)
        self.stereo.setUniquenessRatio(40)
        self.stereo.setTextureThreshold(20)
        logging.info('\nBM Inital Done. (%.2fs)',(time.time() - t0)) #cp3.5
        
    def __del__(self):
        class_name=self.__class__.__name__
        print (class_name,"release")
    
    # @timethis
    def run(self,ImgL,ImgR,Queue,UMat=False):
        t0=time.time()
        ImgL = cv2.cvtColor(ImgL, cv2.COLOR_BGR2GRAY)
        ImgR = cv2.cvtColor(ImgR, cv2.COLOR_BGR2GRAY)
        if UMat:
            disparity = self.stereo.compute(ImgL,ImgR).get().astype(np.float32) / 16.0
        else:
            disparity = self.stereo.compute(ImgL,ImgR).astype(np.float32) / 16.0
        logging.info('\nBM Done. (%.2fs)',(time.time() - t0)) #cp3.5
        Queue.put(disparity)
        # return disparity
        
        
class SGBM:
    """
    @description  : 'SGBM Algorithm Class'
    ---------
    @function  :
    -------
    """  
    count=0
    def __init__(self,cam_mode):
        t0 = time.time()
        SGBM.count += 1
        self.window_size = 3
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=48,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=3,
            P1=8 * 3 * self.window_size ** 2,
            P2=32 * 3 * self.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        logging.info('\nSGBM Inital Done. (%.2fs)',(time.time() - t0)) #cp3.5
    
    def __del__(self):
        class_name=self.__class__.__name__
        print (class_name,"release")
    
    # @timethis
    def run(self,ImgL,ImgR,Queue,UMat=False):
        t0 = time.time()
        if UMat:
            self.disparity = self.stereo.compute(ImgL, ImgR, False).get().astype(np.float32) / 16.0
        else:
            self.disparity = self.stereo.compute(ImgL, ImgR, False).astype(np.float32) / 16.0
        Queue.put(self.disparity)
        return self.disparity

def disparity_centre(raw_box,ratio,disparity,focal,baseline,pixel_size):
    """
    @description  : from disparity map get the depth prediction of the (x_centre,y_centre) point
    ---------
    @param  :
        raw_box: the coordinates of the opposite angle of the prediction box
        ratio: the distance between to centre point
        disparity: type array, disparity map
        focal: focal length in pixel unit 
        baseline: baseline in mm unit
        pixel_size: pixel_size in mm unit
    -------
    @Returns  :
    -------
    """
    '''
    logic: if the pixel number in the box in smaller than 225,than calculate the whole box pixels and get the average, 
    otherwise, 
    '''        
    depth=[]
    #%%%% TODO: 分9个图像框
    # print(raw_box)
    dx,dy=int((raw_box[2]-raw_box[0])*ratio),int((raw_box[3]-raw_box[1])*ratio)
    if (dx == 0) and (dy == 0):
        # %% caculate every pixel in box and get the Median
        for i in range(raw_box[2]-raw_box[0]):
            for j in range(raw_box[3]-raw_box[1]):
                if disparity[(raw_box[0]+i),(raw_box[1]+j)] > 0:
                    depth.append(disparity[(raw_box[0]+i),(raw_box[1]+j)])
    else:
        cx,cy=int((raw_box[0]+raw_box[2])/2),int((raw_box[1]+raw_box[3])/2)
        dw,dh=int((raw_box[2]-raw_box[0])/6),int((raw_box[3]-raw_box[1])/6)
        cxcy=[(cx-2*dw,cy-2*dh),(cx,cy-2*dh),(cx+2*dw,cy-2*dh),\
            (cx-2*dw,cy),(cx,cy),(cx+2*dw,cy),\
            (cx-2*dw,cy+2*dh),(cx,cy+2*dh),(cx+2*dw,cy+2*dh)]
        # print(cxcy)
        # print(dx,dy)    

        #%%%% TODO: 每个框计算深度均值  
        for x_centre,y_centre in cxcy:
            p=[-2,-1,0,1,2]
            d=np.zeros((25,),dtype=float)
            dis_mean=0.
            for i in range(5):
                for j in range(5):
                    nx,ny=int(x_centre+p[i]*dx),int(y_centre+p[j]*dy)
                    # print('(%d,%d)'%(nx,ny),end=' ')
                    d.flat[5*i+j]=disparity[ny,nx]
            d=d.ravel()
            d=d[d>0.]
            d=np.sort(d,axis=None)
            # print(d)
            if len(d) >= 5:
                d=np.delete(d,[0,-1])
                dis_mean = d.mean()
                depth.append(dis_mean)
    # %%%% TODO: 取众多框计算值的中位数 
    # print('depth: ',depth)
    depth.sort()
    if len(depth) == 0:
        temp_dis = -1
    elif (len(depth)%2 == 0) & (len(depth)>1):
        temp_dis = ((focal*baseline/depth[math.floor(len(depth)/2)])+(focal*baseline/depth[math.floor(len(depth)/2)-1]))/2
    else:
        temp_dis = focal*baseline/depth[math.floor(len(depth)/2)]
    # print(temp_dis)
    return temp_dis

# %% standalone usage function
def stereo_sgbm(ImgLPath='../data/images/left.png',ImgRPath='../data/images/right.png', path=True):
    t0 = time.time()
    imgL = cv2.imread(ImgLPath)
    imgR = cv2.imread(ImgRPath)
    # logging.info(f'Images Inital Done. ({time.time() - t0:.3f}s)') #cp3.6
    logging.info('Images Inital Done. (%.2fs)',(time.time() - t0)) #cp3.5
    # disparity range tuning
    window_size = 3
    # min_disp = 0
    # num_disp = 320 - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=3,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    # logging.info(f'SGBM Done. ({time.time() - t0:.3f}s)') #cp3.6
    logging.info('SGBM Done. (%.2fs)',(time.time() - t0)) #cp3.5
    print('SGBM Done. (%.2fs)'%(time.time() - t0)) #cp3.5
    return disparity

def detect_disparity(ImgLPath='../data/images/Left1_rectified.bmp',ImgRPath='../data/images/Right1_rectified.bmp'):
    imgL = cv2.imread(ImgLPath)
    imgR = cv2.imread(ImgRPath)
    # disparity range tuning
    window_size = 3
    # min_disp = 0
    # num_disp = 320 - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=3,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    fig, ax = plt.subplots()
    plt.imshow(disparity, 'gray')
    cursor = Cursor(ax)
    fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_source', type=str, default='data/images/left.png', help='source')
    parser.add_argument('--right_source', type=str, default='data/images/left.png', help='source')
    opt = parser.parse_args()
    print(opt)    
    detect_disparity()


