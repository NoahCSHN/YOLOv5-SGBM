#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:08:03 2021

@author: bynav
"""

import numpy as np
import cv2
from utils.general import timethis,calib_type
 
####################仅仅是一个示例###################################
 
 
# 双目相机参数
class stereoCamera(object):
    # @timethis
    def __init__(self,mode=1,height=1280,width=960):
        if mode==calib_type.OV9714_1280_720:
            # %% OV9714 1280x720
            # 左相机内参
            print('Camera OV9714 1280X720')
            self.cam_matrix_left = np.array([[1147.039625617031, 0, 733.098961811485],
                                            [0., 1150.152056805138, 380.840107187423],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[1143.705662745462, 0, 708.710387078841],
                                            [0., 1145.008623150225, 378.384254182898],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[-0.008570670886844, 0.205372331430449, 0.000333267442408, 0.013776316839033, -0.568892690808155]])
            self.distortion_r = np.array([[0.017680717249695, 0.063462410840028, -0.000293093935331, 0.013855048079951, -0.189246079109404]])
    
            # 旋转矩阵
            self.R = np.array([[0.999999732264347, -0.000596922024167, 0.000423267446915],
                            [0.000597153117432,  0.999999672614051, -0.000546058554318],
                            [-0.000422941353966, 0.000546311163595, 0.999999761332333]])
    
            # 平移矩阵
            self.T = np.array([[-59.923725127564538], [0.009422096967625], [-2.535572114734797]])
    
            # 焦距 unit:pixel resolution ratio ，1280*720 1191.47 640*640 595.735 416*416 387.23   3.6mm
            self.focal_length = 387.23

            # 焦距 unit:pixel 949.62
            self.focal_length_pix = 1191.47  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 59.923725127564538  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00300
        elif mode==calib_type.AR0135_1280_720:
            # %% AR0135 1280x720
            # 左相机内参
            print('Camera AR0135 1280x720')
            self.cam_matrix_left = np.array([[899.9306, 0, 672.5951],
                                            [0., 903.0798, 472.7901],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[901.7015, 0, 670.4376],
                                            [0., 904.3837, 495.7119],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[0.0124, 0.0010, 0.0000, 0.0064, -0.0902]])
            self.distortion_r = np.array([[0.0112, -0.0053, -0.0006, 0.0070, -0.0457]])
    
            # 旋转矩阵
            self.R = np.array([[0.999999544012909, -0.000361332677693, -0.000883975491791],
                            [0.000358586486588, 0.999995115724826, -0.003104825634915],
                            [0.000885093049172, 0.003104507237489, 0.999994789308978]])
    
            # 平移矩阵
            self.T = np.array([[-44.789005274626312], [0.369693318112806], [-1.106236740447207]])
    
            # 焦距 unit:pixel resolution ratio ，1280*720 932.83 640*640 466.415 416*416 303.17
            self.focal_length = 303.17

            # 焦距 unit:pixel 949.62
            self.focal_length_pix = 932.83  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 44.789005274626312  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00375
        else:
            # %% 1280x960
            # 左相机内参
            print('Camera AR0135 1280x960')
            self.cam_matrix_left = np.array([[900.114185349224, 0, 670.075287577187],
                                            [0., 904.228316085958, 471.765023773669],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[896.135584664394, 0, 669.702575099279],
                                            [0., 899.168992319405, 492.994037592607],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[0.00488906840212177, -0.0161131150686868, -0.00258596027519168, 0.00526371337538908, -0.0418911874416988]])
            self.distortion_r = np.array([[-0.013904915007985, 0.0516707668013402, -0.00178988171280563, 0.00648086228771938, -0.0918755222306161]])
    
            # 旋转矩阵
            self.R = np.array([[0.999997343532194, -0.000356518914434726, -0.00227724017584731],
                            [0.000356153825781219, 0.999999923661093, -0.000160724174948289],
                            [0.00227729730321366, 0.000159912700188857, 0.999997394169065]])
    
            # 平移矩阵
            self.T = np.array([[-44.9047784155448], [0.403175791582689], [-1.1491319659264]])
    
            # 焦距 unit:pixel resolution ratio ，1280*960 928.778 640*640 464.389 416*416 344.906
            self.focal_length = 301.852849992

            # 焦距 unit:pixel 928.778
            self.focal_length_pix = 928.778  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 44.9047784155448  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00375


        # 计算校正变换
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(self.cam_matrix_left, self.distortion_l, self.cam_matrix_right,\
                                                                                             self.distortion_r, (width, height), self.R, self.T, alpha=0)
    
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cam_matrix_left, self.distortion_l, self.R1, self.P1, (width, height), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cam_matrix_right, self.distortion_r, self.R2, self.P2, (width, height), cv2.CV_32FC1)
        print(self.Q)