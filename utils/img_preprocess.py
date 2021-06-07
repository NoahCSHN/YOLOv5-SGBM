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
from utils.general import timethis,timeblock,calib_type,letterbox
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
 
# @timethis
def Image_Rectification(camera_config, img_left, img_right, imgsz=640, path=False, debug=False, UMat=False, cam_mode=4):
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
        cam_mode: type int, the camera configuration type
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
    # 读取相机内参和外参
    config = camera_config        
    if cam_mode == calib_type.AR0135_416_416.value or cam_mode == calib_type.AR0135_640_640.value:
        if UMat:
            img_raw = cv2.UMat.get(iml)
        else:
            img_raw = iml
        iml, gain, padding = letterbox(iml, imgsz)
        imr = letterbox(imr, imgsz)[0]
        # 立体校正
        img_ai, imr_rectified = rectifyImage(iml, imr, config.map1x, config.map1y, config.map2x, config.map2y)   
    elif cam_mode == calib_type.MIDDLEBURY_416.value:
        img_ai, gain, padding = letterbox(iml, imgsz)
        imr_rectified = letterbox(imr, imgsz)[0]
        img_raw = iml
    else:
        # 立体校正
        iml_rectified, imr_rectified = rectifyImage(iml, imr, config.map1x, config.map1y, config.map2x, config.map2y)
        if UMat:
            img_raw = cv2.UMat.get(iml_rectified)
        else:
            img_raw = iml_rectified
        # 图像缩放
        img_ai, gain, padding = letterbox(iml_rectified, imgsz)
        imr_rectified = letterbox(imr_rectified, imgsz)[0]
    # save for debug
    # cv2.imwrite('./runs/detect/test/Left1_rectified.bmp', iml_rectified)
    # cv2.imwrite('./runs/detect/test/Right1_rectified.bmp', imr_rectified)
 
    if debug:
    # 绘制等间距平行线，检查立体校正的效果
        line = draw_line(img_ai, imr_rectified)
        cv2.imwrite('./runs/detect/test/line.png', line)
    iml_rectified = cv2.cvtColor(img_ai, cv2.COLOR_BGR2GRAY)
    imr_rectified = cv2.cvtColor(imr_rectified, cv2.COLOR_BGR2GRAY)        
    iml_rectified = np.ascontiguousarray(iml_rectified)
    imr_rectified = np.ascontiguousarray(imr_rectified)
 
    return img_raw, img_ai, iml_rectified, imr_rectified, gain, padding

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
