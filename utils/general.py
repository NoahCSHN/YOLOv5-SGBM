'''
Author  : Noah
Date    : 20210408
function: common-basic function database
'''
import os,sys,logging,cv2,time
from pathlib import Path
import numpy as np
from contextlib import contextmanager

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
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
    # pwd = os.getcwd()
    # real_root = os.path.abspath(root_path)
    # relpath = os.path.relpath(pwd,root_path)

    path = os.path.join(root_path,new_path)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper

@contextmanager
def timeblock(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('{} : {}'.format(label, end - start))