#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : 将左右相机拍摄的两张照片合成一张
@Date     :2021/05/28 19:38:05
@Author      :Yyc
@version      :1.0
'''

import os,time,sys,cv2
import numpy as np

#%% 
#TODO:打开两张照片，合成一张
def merge_images(root='',folder=''):
    # print(folder)
    dir = os.path.join(root,folder)
    for _,_,files in os.walk(dir,topdown=False):
        # print(files)
        if 'im0.png' in files:
            filel = os.path.join(dir,'im0.png')
            iml = cv2.imread(filel)
        else:
            print('Image Left not found')
            return -1
        if 'im1.png' in files:
            filer = os.path.join(dir,'im1.png')
            imr = cv2.imread(filer)
        else:
            print('Image Right not found')
            return -1
        # width = 2*imgl.shape[1]
        # height = imgl.shape[0]
        im_merge = cv2.hconcat([iml,imr])
        save_path = os.path.join(root,'merge')
        print(save_path)
        cv2.imshow('merge image',im_merge)
        cv2.imwrite(os.path.join(save_path,folder+'.png'),im_merge)
        if cv2.waitKey(1) == ord('q'):
            raise KeyboardInterrupt
    

#%%
#TODO：遍历指定文件夹下的所有文件夹，搜索其中的im0.png，im1.png文件
def parser_folder(parent_folder=''):
    for root,folders,_ in os.walk(parent_folder,topdown=False):
        for folder in folders:
            merge_images(root,folder)

#%%
#main:
if __name__ == '__main__':
    parser_folder('/home/bynav/RK3399/AI_SGBM/data/stereo')