#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:06:14 2021

@author: bynav
"""

# -*- coding: utf-8 -*-
import cv2,time,logging
import numpy as np
from utils.stereoconfig import stereoCamera
import os
from pathlib import Path
from utils.rknn_detect_yolov5 import letterbox
from utils.general import timethis
# from pcl import pcl_visualization
 

# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if(img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
 
    return img1, img2

 
# 消除畸变
# @timethis
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
 
    return undistortion_image
 
 
# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
# @timethis
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T
 
    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=0)
 
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
 
    return map1x, map1y, map2x, map2y, Q
 
 
# 畸变校正和立体校正
# @timethis
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
 
    return rectifyed_img1, rectifyed_img2
 
 
# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
 
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
 
    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
 
    return output

def resize_convert(imgl_rectified, imgr_rectified, imgsz=640, stride=32):
    imgl_rectified = letterbox(imgl_rectified, imgsz)[0]
    imgr_rectified = letterbox(imgr_rectified, imgsz)[0]
    
    # Convert
    img_ai = imgl_rectified[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img_ai = np.ascontiguousarray(img_ai)
    imgl_rectified = np.ascontiguousarray(imgl_rectified)
    imgr_rectified = np.ascontiguousarray(imgr_rectified)
    return img_ai, imgl_rectified, imgr_rectified
 
@timethis
def Image_Rectification(camera_config, img_left, img_right, imgsz=640, path=False, debug=False, UMat=False):
    """
    @description  : stereo camera calibration
    ---------
    @param  : 
        camera_config: type class stereoCamera, stereo camera calibration parameters
        img_left: type array, image from the left camera
        img_right: type array, image from the right camera
        imgsz: type tuple (int,int), used by Image_Rectification function resize images for SGBM to the same size as image for object detection
        path: type bool, if true, the img_left and img_right type are string,as image file path 
        debug: type bool, reserved
        UMat: type bool, if true, use GPU accelarating SGBM and image rectification process
    -------
    @Returns  :
        iml_rectified: type array, left image for SGBM
        imr_rectified: type array, right image for SGBM
        img_ai_raw: type array, image for object detection
        (height,width): type tuple(int,int), reserved
    -------
    """
    
    
    # 读取MiddleBurry数据集的图片
    t0 = time.time()
    if path:
        imgl_path=str(Path(img_left).absolute())
        imgr_path=str(Path(img_right).absolute())
        iml = cv2.imread(imgl_path)  # left
        imr = cv2.imread(imgr_path)  # right
    else:
        iml = img_left  # left
        imr = img_right # right
    if UMat:
        iml = cv2.UMat(iml)  # left
        imr = cv2.UMat(imr) # right        
        height, width = iml.get().shape[0:2]
    else:
        height, width = iml.shape[0:2]
    
    # 读取相机内参和外参
    config = camera_config
 
    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    if UMat:
        img_ai_raw = cv2.UMat.get(iml_rectified)
    else:
        img_ai_raw = iml_rectified
    # 图像缩放
    iml_rectified = letterbox(iml_rectified, imgsz)[0]
    imr_rectified = letterbox(imr_rectified, imgsz)[0]
    # save for debug
    # cv2.imwrite('/home/bynav/AI_SGBM/runs/detect/exp/Left1_rectified.bmp', iml_rectified)
    # cv2.imwrite('/home/bynav/AI_SGBM/runs/detect/exp/Right1_rectified.bmp', imr_rectified)
    # print(Q)
 
    # if debug:
    # 绘制等间距平行线，检查立体校正的效果
        # line = draw_line(iml_rectified, imr_rectified)
        # cv2.imwrite('./runs/detect/test/line.png', line)
 
    # 显示点云
    # view_cloud(pointcloud()
    # logging.info(f'Image rectification Done. ({time.time() - t0:.3f}s)')   #cp3.6
    logging.info('Image rectification Done. (%.2fs)',(time.time() - t0))   #cp3.5
    # print('Image rectification Done. (%.2fs)'%(time.time() - t0))   #cp3.5
    return iml_rectified,imr_rectified,img_ai_raw,(height,width)

if __name__ == '__main__':
    config = stereoCamera()
    img_left = '../data/images/Left1.bmp'
    img_right = '../data/images/Right1.bmp'
    left,right,left_rgb = Image_Rectification(config, img_left, img_right, path=True)
    cv2.imshow('left',left)
    cv2.waitKey(500)
    cv2.imshow('right',right)
    cv2.waitKey(500)
    cv2.imshow('left_rgb',left_rgb)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()