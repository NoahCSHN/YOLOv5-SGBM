#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:08:03 2021

@author: bynav
"""

import numpy as np
 
 
####################仅仅是一个示例###################################
 
 
# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # %% OV9714 1280x720
        # 左相机内参
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
        
        '''
        # %% AR0135 1280x960
        # 左相机内参
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
        '''
        ''' 
        # %% 1280x720
        # 左相机内参
        self.cam_matrix_left = np.array([[906.052143341671, 0, 662.035976643843],
                                         [0., 909.810278518771, 347.812877929693],
                                         [0., 0., 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[904.039816599602, 0, 657.374726861859],
                                          [0., 907.144037650865, 371.505562447014],
                                          [0., 0., 1.]])
 
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.00069451800687876, 0.0526241964124028, -0.00395258899740901, 0.0029474338458693, -0.135921756945649]])
        self.distortion_r = np.array([[0.00503605508798055, 0.0137569348762201, -0.00339148012519355, 0.00274948038873249, -0.0691817876373664]])
 
        # 旋转矩阵
        self.R = np.array([[0.999999944353214, 0.000141559982260522, 0.00030208333489573],
                           [-0.000141022307976901, 0.999998407337637, -0.00177916691099839],
                           [-0.000302334712615328, 0.00177912421150438, 0.999998371654055]])
 
        # 平移矩阵
        self.T = np.array([[-44.7734208533592], [0.329530613742665], [-0.453620226449132]])
 
        # 焦距 unit:pixel resolution ratio ，1280*720 949.62 640*640 474.81 416*416 316.54
        self.focal_length = 316.54  

        # 焦距 unit:pixel 949.62
        self.focal_length_pix = 949.62  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = 44.7734208533592  # 单位：mm， 为平移向量的第一个参数（取绝对值）
        
        # pixel size unit: mm
        self.pixel_size = 0.00375
        '''