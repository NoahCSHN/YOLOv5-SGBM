'''
Author  : Noah
Date    : 20210412
function: generate dataset for rknn quantization
'''
import os,sys,glob,time,random,argparse
from pathlib import Path
image_format = ['.jpg','.png']

if __name__ == '__main__':
    # %% 
    # parameter input with model start up
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="The image source folder path", type=str, default='../yolo_easy/data/images/left')
    parser.add_argument("--output", help="output path", type=str, default='../yolo_sololife')
    parser.add_argument("--num", help="the image number for dataset generate", type=int, default=500)
    # parser.add_argument("--device", help="device on which model runs", type=str,default='pc')
    # parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    # parser.add_argument("--webcam", help="connect to real camera", action="store_true")
    # parser.add_argument("--BM", help="switch to BM alogrithm for depth inference", action="store_true")                    
    # parser.add_argument("--debug", help="run under debug mode", action="store_true")
    # parser.add_argument("--save_path",help="path for result saving",type=str,default="runs/detect/test")                    
    args = parser.parse_args()
    print(args)    

    # %%
    # read images in folder and do random sort
    path = os.path.abspath(args.source)
    assert os.path.isdir(path),'%s is not a existed folder' %path

    files = next(os.walk(path))[2]
    images = []
    for file in files:
        if os.path.splitext(file)[1] in image_format:
            images.append(file)
    assert len(images) >= args.num,'There are not enough images in the source folder'
    random.shuffle(images)

    # %%
    # get the first num images path and write it in dataset.txt
    assert os.path.isdir(args.output),'the output folder is not existed' 
    with open(os.path.join(args.output,'dataset.txt'),'w') as f:
        i = 0
        for image in images:
            if i >= args.num:
                break
            file_index=os.path.join(path,image)
            f.write(file_index)
            f.write('\n')
            i+=1
