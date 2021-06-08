import numpy as np
import argparse,logging,time,math,os
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

class Stereo_Matching:
    """
    @description  : Stereo_Matching Algorithm class
    ---------
    @function  :
    -------
    """
    count=0
    def __init__(self,cam_mode,BM=False,filter=True, 
                 filter_lambda=8000.0,filter_sigma=1.5,
                 filter_unira=5,
                 numdisparity=64, mindis=0, block=9, TextureThreshold=5,
                 prefiltercap=63, prefiltersize=9, prefiltertype=1,
                 SpeckleWindowSize=50, speckleRange=2,
                 sf_path=''):
        """
        @description  : initialize stereoBM or stereoSGBM alogrithm
        ---------
        @param  : cam_mode: reserved
        @param  : BM: bool, if false, use stereoSGBM, otherwise, use stereoBM
        @param  : filter: bool, if false, use stereoSGBM, otherwise, use stereoBM
        @param  : filter_lambda: float, the lambda parameter of post WLS filter
        @param  : filter_sigma: float, the sigmacolor parameter of post WLS filter
        @param  : filter_unira: int, Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
        @param  : numdisparity: int, Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
        @param  : mindis: int, Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
        @param  : block: int, Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        @param  : TextureThreshold: int, must be dividable by 16, the min disparity the SGBM will attempt
        @param  : prefiltercap: int, Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
        @param  : prefiltersize: int, Pre-processing filter window size, allowed range is [5,255], generally should be between 5x5 ... 21x21, parameters must be odd, INT type
        @param  : prefiltertype: int, he type of preparation filter is mainly used to reduce the photometric distortions, eliminate noise and enhanced texture, etc., there are two optional types: CV_STEREO_BM_NORMALIZED_RESPONSE (normalized response) or CV_STEREO_BM_XSOBEL (horizontal direction Sobel Operator) , Default type), this parameter is int type
        @param  : SpeckleWindowSize: int, Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        @param  : speckleRange: int, Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
        @param  : sf_path: str, the stereo algorithm configuration save path
        -------
        @Returns  :
        -------
        """
        t0 = time.time()
        self.BM = BM
        Stereo_Matching.count += 1
        self.filter_en = filter
        self.lamdba=filter_lambda
        self.sigma=filter_sigma
        self.unira=filter_unira
        if not self.BM:
            self.window_size = 3
            '''
            #The second parameter controlling the disparity smoothness. 
            # The larger the values are, the smoother the disparity is. 
            # P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. 
            # P2 is the penalty on the disparity change by more than 1 between neighbor pixels. 
            # The algorithm requires P2 > P1 . 
            # See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*blockSize*blockSize and 32*number_of_image_channels*blockSize*blockSize , respectively).
            '''            
            self.left_matcher = cv2.StereoSGBM_create(
                minDisparity=mindis,
                numDisparities=numdisparity-mindis,  # max_disp has to be dividable by 16 f. E. HH 192, 256
                blockSize=block,
                P1=8 * 3 * self.window_size ** 2,   
                P2=32 * 3 * self.window_size ** 2,  
                disp12MaxDiff=1,
                uniquenessRatio=self.unira,
                speckleWindowSize=SpeckleWindowSize,
                speckleRange=speckleRange,
                preFilterCap=prefiltercap,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )
            print('\nSGBM Initial Done. (%.2fs)'%(time.time() - t0)) #cp3.5
        else:
            self.left_matcher = cv2.StereoBM_create(numdisparity, block)
            self.left_matcher.setUniquenessRatio(self.unira)
            self.left_matcher.setTextureThreshold(TextureThreshold)
            self.left_matcher.setMinDisparity(mindis)
            self.left_matcher.setDisp12MaxDiff(1)
            self.left_matcher.setSpeckleRange(speckleRange)
            self.left_matcher.setSpeckleWindowSize(SpeckleWindowSize)
            self.left_matcher.setBlockSize(block)
            self.left_matcher.setNumDisparities(numdisparity)
            self.left_matcher.setPreFilterCap(prefiltercap)
            self.left_matcher.setPreFilterSize(prefiltersize)
            self.left_matcher.setPreFilterType(prefiltertype)
            # self.left_matcher.setROI1(0)
            # self.left_matcher.setROI2(0)
            # self.left_matcher.setSmallerBlockSize(0)

            print('\nBM Initial Done. (%.2fs)'%(time.time() - t0)) #cp3.5
        if self.filter_en:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.filter.setLambda(self.lamdba)
            self.filter.setSigmaColor(self.sigma)
        if sf_path != '':
            self.write_file(sf_path)

    def change_parameters(self,filter_unira=-1,filter_lambda=-1,filter_sigma=-1):
        """
        @description  : reserved, TODO: dynamic adjust filter parameters
        ---------
        @param  : filter_lambda, float, the lambda parameter of post WLS filter
        @param  : filter_sigma, float, the sigmacolor parameter of post WLS filter
        @param  : filter_unira, int, the UniquenessRatio parameter of the stereo alogrithm filter
        -------
        @Returns  : -1 ,error, 0, normal
        -------
        """        
        if Stereo_Matching.count == 0:
            print ('No Stereo Matching instance find.')
            return -1
        if filter_unira >= 0:
            self.stereo.setUniquenessRatio(filter_unira)
            print('set UniquenessRatio: %d'%filter_unira)
        if filter_lambda >= 0:
            self.filter.setLambda(filter_lambda)
            print('set filter Lambda: %f'%filter_lambda)
        if filter_sigma >= 0:
            self.filter.setSigmaColor(filter_sigma)
            print('set filter sigma: %f'%filter_sigma)
        return 0

    def write_file(self,path):
        self.sf = cv2.FileStorage()
        file_path = os.path.join(path,'stereo_config.xml')
        self.sf.open(file_path,cv2.FileStorage_WRITE)
        self.sf.write('datetime', time.asctime())
        if self.BM:
            self.sf.startWriteStruct('stereoBM',cv2.FileNode_MAP)
        else:
            self.sf.startWriteStruct('stereoSGBM',cv2.FileNode_MAP)
        self.sf.write('NumDisparities',self.left_matcher.getNumDisparities())
        self.sf.write('MinDisparity',self.left_matcher.getMinDisparity())
        self.sf.write('BlockSize',self.left_matcher.getBlockSize())
        self.sf.write('Disp12MaxDiff',self.left_matcher.getDisp12MaxDiff())
        self.sf.write('SpeckleRange',self.left_matcher.getSpeckleRange())
        self.sf.write('SpeckleWindowSize',self.left_matcher.getSpeckleWindowSize())
        self.sf.write('PreFilterCap',self.left_matcher.getPreFilterCap())
        self.sf.write('UniquenessRatio',self.left_matcher.getUniquenessRatio())
        if self.BM:
            self.sf.write('PreFilterSize',self.left_matcher.getPreFilterSize())
            self.sf.write('PreFilterType',self.left_matcher.getPreFilterType())
            self.sf.write('ROI1',self.left_matcher.getROI1())
            self.sf.write('ROI2',self.left_matcher.getROI2())
            self.sf.write('SmallerBlockSize',self.left_matcher.getSmallerBlockSize())
            self.sf.write('TextureThreshold',self.left_matcher.getTextureThreshold())
        else:
            self.sf.write('Mode',self.left_matcher.getMode())
        self.sf.endWriteStruct()
        if self.filter_en:
            self.sf.startWriteStruct('DisparityWLSFilter',cv2.FileNode_MAP)
            self.sf.write('ConfidenceMap',self.filter.getConfidenceMap())
            self.sf.write('DepthDiscontinuityRadius',self.filter.getDepthDiscontinuityRadius())
            self.sf.write('Lambda',self.filter.getLambda())
            self.sf.write('LRCthresh',self.filter.getLRCthresh())
            self.sf.write('ROI',self.filter.getROI())
            self.sf.write('SigmaColor',self.filter.getSigmaColor())
            self.sf.endWriteStruct()
        self.sf.release()

    def __del__(self):
        class_name=self.__class__.__name__
        print ('\n',class_name,"release")
    
    # @timethis
    def run(self,ImgL,ImgR,Q,Queue,UMat=False,filter=True):
        """
        @description  :compute the disparity of ImgL and ImgR and put the disparity map to Queue
        ---------
        @param  : ImgL, Gray image taked by the left camera
        @param  : ImgR, Gray image taked by the right camera
        @param  : Queue, the data container of python API queue, used for data interaction between thread
        @param  : UMat, bool, if true, the data type is UMat(GPU), otherwise, the data type is UMat(CPU)
        @param  : filter, bool, if true, return the disparity map with post filter, otherwise, return the raw disparity map
        -------
        @Returns  : disparity, a Mat with the same shape as ImgL
        -------
        """
        t0=time.time()
        if not self.filter_en:
            if not self.BM:
                if UMat:
                    disparity_left = self.left_matcher.compute(ImgL, ImgR, False).get().astype(np.float32) / 16.0
                else:
                    disparity_left = self.left_matcher.compute(ImgL, ImgR, False).astype(np.float32) / 16.0
            else:
                if UMat:
                    disparity_left = self.left_matcher.compute(ImgL,ImgR).get().astype(np.float32) / 16.0
                else:
                    disparity_left = self.left_matcher.compute(ImgL,ImgR).astype(np.float32) / 16.0
                logging.info('\nBM Done. (%.2fs)',(time.time() - t0)) #cp3.5  
            color_3d = cv2.reprojectImageTo3D(disparity_left,Q).reshape(-1,416,3)
            Queue.put((disparity_left,color_3d))
        else:
            if not self.BM:
                if UMat:
                    disparity_left = self.left_matcher.compute(ImgL, ImgR, False).get().astype(np.float32) / 16.0
                    disparity_right = self.right_matcher.compute(ImgR, ImgL, False).get().astype(np.float32) / 16.0
                else:
                    disparity_left = self.left_matcher.compute(ImgL, ImgR, False).astype(np.float32) / 16.0
                    disparity_right = self.right_matcher.compute(ImgR, ImgL, False).astype(np.float32) / 16.0
            else:
                if UMat:
                    disparity_left = self.left_matcher.compute(ImgL,ImgR).get().astype(np.float32) / 16.0
                    disparity_right = self.right_matcher.compute(ImgR, ImgL).get().astype(np.float32) / 16.0
                else:
                    disparity_left = self.left_matcher.compute(ImgL,ImgR).astype(np.float32) / 16.0
                    disparity_right = self.right_matcher.compute(ImgR, ImgL).astype(np.float32) / 16.0
                logging.info('\nBM Done. (%.2fs)',(time.time() - t0)) #cp3.5            
            disparity=self.filter.filter(disparity_left, ImgL, disparity_map_right=disparity_right)
            color_3d = cv2.reprojectImageTo3D(disparity,Q).reshape(-1,416,3)
            Queue.put((disparity,color_3d))

        

def disparity_centre(raw_box,ratio,disparity,depth_map,focal,baseline,pixel_size):
    """
    @description  : from disparity map get the depth prediction of the (x_centre,y_centre) point
    ---------
    @param  :
        raw_box: the coordinates of the opposite angle of the prediction box
        ratio: the distance between to centre point
        disparity: type array, disparity map
        depth_map: type array, depth map
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
            print('\ndisparity row:',end=' ')
            for j in range(raw_box[3]-raw_box[1]):
                print(disparity[(raw_box[0]+i),(raw_box[1]+j)],end=',')
                # if disparity[(raw_box[0]+i),(raw_box[1]+j)] > -11:
                #     depth.append(disparity[(raw_box[0]+i),(raw_box[1]+j)])
                depth.append(depth_map[(raw_box[0]+i),(raw_box[1]+j)])
        print(depth,end='\r')
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
                    # d.flat[5*i+j]=disparity[ny,nx]
                    d.flat[5*i+j]=depth_map[ny,nx]
            d=d.ravel()
            d=d[d>-11.]
            d=np.sort(d,axis=None)
            print(d,end='\r')
            if len(d) >= 5:
                d=np.delete(d,[0,-1])
                dis_mean = d.mean()
                depth.append(dis_mean)
    # %%%% TODO: 取众多框计算值的中位数 
    depth = np.abs(depth)
    depth.sort()
    if len(depth) == 0:
        temp_dis = -1
    elif (len(depth)%2 == 0) & (len(depth)>1):
        if (depth[math.floor(len(depth)/2)] != 0) and (depth[math.floor(len(depth)/2)-1] != 0):
            # temp_dis = ((focal*baseline/abs(depth[math.floor(len(depth)/2)]))+(focal*baseline/abs(depth[math.floor(len(depth)/2)-1])))/2
            temp_dis = (depth[math.floor(len(depth)/2)] + depth[math.floor(len(depth)/2)-1])/2
        else:
            temp_dis = -1
    else:
        if depth[math.floor(len(depth)/2)] != 0:
            # temp_dis = focal*baseline/abs(depth[math.floor(len(depth)/2)])
            temp_dis = depth[math.floor(len(depth)/2)]
        else:
            temp_dis = -1
    return temp_dis

def remove_invalid(disp_arr, points, colors):
    mask = (
        (disp_arr > disp_arr.min()) &
        np.all(~np.isnan(points), axis=1) &
        np.all(~np.isinf(points), axis=1)
    )
    return points[mask], colors[mask]


def calc_point_cloud(image, disp, q):
    points = cv2.reprojectImageTo3D(disp, q).reshape(-1, 3)
    colors = image.reshape(-1, 3)
    return remove_invalid(disp.reshape(-1), points, colors)


def project_points(points, colors, r, t, k, dist_coeff, width, height):
    projected, _ = cv2.projectPoints(points, r, t, k, dist_coeff)
    xy = projected.reshape(-1, 2).astype(np.int)
    mask = (
        (0 <= xy[:, 0]) & (xy[:, 0] < width) &
        (0 <= xy[:, 1]) & (xy[:, 1] < height)
    )
    return xy[mask], colors[mask]


def calc_projected_image(points, colors, r, t, k, dist_coeff, width, height):
    xy, cm = project_points(points, colors, r, t, k, dist_coeff, width, height)
    image = np.zeros((height, width, 3), dtype=colors.dtype)
    image[xy[:, 1], xy[:, 0]] = cm
    return image


def rotate(arr, anglex, anglez):
    return np.array([  # rx
        [1, 0, 0],
        [0, np.cos(anglex), -np.sin(anglex)],
        [0, np.sin(anglex), np.cos(anglex)]
    ]).dot(np.array([  # rz
        [np.cos(anglez), 0, np.sin(anglez)],
        [0, 1, 0],
        [-np.sin(anglez), 0, np.cos(anglez)]
    ])).dot(arr)


def reproject_3dcloud(left_image, disparity, focal_length, tx):
    image = left_image
    height, width, _ = image.shape

    q = np.array([
        [1, 0, 0, -width/2],
        [0, 1, 0, -height/2],
        [0, 0, 0, focal_length],
        [0, 0, -1/tx, 0]
    ])
    points, colors = calc_point_cloud(image, disparity, q)

    r = np.eye(3)
    t = np.array([0, 0, -100.0])
    k = np.array([
        [focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]
    ])
    dist_coeff = np.zeros((4, 1))

    def view(r, t):
        cv2.imshow('projected', calc_projected_image(
            points, colors, r, t, k, dist_coeff, width, height
        ))

    view(r, t)

    angles = {  # x, z
        'w': (-np.pi/6, 0),
        's': (np.pi/6, 0),
        'a': (0, np.pi/6),
        'd': (0, -np.pi/6)
    }

    while 1:
        key = cv2.waitKey(0)

        if key not in range(256):
            continue

        ch = chr(key)
        if ch in angles:
            ax, az = angles[ch]
            r = rotate(r, -ax, -az)
            t = rotate(t, ax, az)
            view(r, t)

        elif ch == '\x1b':  # esc
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_source', type=str, default='data/images/left.png', help='source')
    parser.add_argument('--right_source', type=str, default='data/images/left.png', help='source')
    opt = parser.parse_args()
    print(opt)    


