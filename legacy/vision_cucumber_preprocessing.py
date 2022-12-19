#!/usr/bin/env python
# coding: utf-8

# In[1]:


########################################################
# import libraries 
########################################################

########################################################
## To get RGB & Depth image from Realsense
    ### Common libraries are deleted
########################################################

from numpy.core.numeric import NaN
import pyrealsense2 as rs
import time
from datetime import datetime

########################################################
## To detect target objects with trained object detection model(YOLOR)
    ### Common libraries are deleted
########################################################

import argparse
import platform
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

# from utils.google_utils import attempt_load
# from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *
from math import pi


########################################################
## To process image to get growth information
########################################################

import os # to open folder with directory
import cv2 # to use OpenCV
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics as stat
import matplotlib

from time import sleep 
from pandas import read_csv
from PIL import Image # to open image file
from decimal import Decimal # to convert str to double
from IPython.display import Image

from tkinter import filedialog
from tkinter import *

def showme(img):
    cv2.imshow("", img)
#     cv2.namedWindow("", flags = cv2.WINDOW_NORMAL)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[2]:



########################################################
## To process image to get growth information
########################################################

import os # to open folder with directory
import cv2 # to use OpenCV
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics as stat
import matplotlib

from time import sleep 
from pandas import read_csv
from PIL import Image # to open image file
from decimal import Decimal # to convert str to double
from IPython.display import Image

## function for see the image in jupyter notebook
# def showme(img):
#     cv2.namedWindow("", flags = cv2.WINDOW_NORMAL)
#     cv2.imshow("", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

### 2. Model Inference Part


# In[10]:



### 1. Variables for Inference ###
DEVICE = ""
OUTPUT = "inference/output"
CFG = "cfg/yolor_p6.cfg"
# CFG = "cfg/yolov4.cfg"




IMGSZ = 1280
WEIGHTS = ["weights/yolor/yolor_p611/weights/best_overall.pt"]
NAMES = "data/farmbot_cucumber.names"
    # MODEL_TYPE = "rgb"
# 변경
   
# m = int(input("MODEL_TYPE 입력 : \nsubtr = 1\nblur21 = 2\nblur100 = 3\n blur9 = 4\noriginal rgb = 5\ngreyAbs = 6\n"))

# if m == 1:
#     MODEL_TYPE = "subtr"
#     print(MODEL_TYPE)
# elif m == 2:
#     MODEL_TYPE = "blur21"
# elif m == 3:
#     MODEL_TYPE = "blur100"
# elif m == 4:
#     MODEL_TYPE = "blur9"
# elif m == 5:
#     MODEL_TYPE = "rgb"
# elif m == 6:
#     MODEL_TYPE = "greyAbs"

# print("Processing :",MODEL_TYPE)

AUGMENT = False
CLASSES = None
AGNOSTIC_NMS = False
UPDATE = False
save_txt = True

CONF_THRES = 0.2
# CONF_THRES = 0.5
IOU_THRES = 0.5
low_rate = 0.2
high_rate = 0.5
numW = 1280
numH = 720

eps = 10 ** -6

stemLine_color = (208,205,195)
stemROILine_color = (0,255,204)
plantLine_color = (84, 230, 129)

cnt = 0 # count for input n
n = 0 # input n

### 2. Model Inference & Functions

def sqrt(N):
    return N ** (1/2)

def morphology(depth):
    depth = cv2.merge((depth,depth,depth))
    depth = np.array(depth,dtype = np.uint8)

    kernel = np.ones((5,5), np.uint8)

    depth_median = depth.copy()

    for i in range(10):
        depth_median = cv2.medianBlur(depth_median,5)

    depth_dilate = cv2.dilate(depth_median,kernel,iterations = 2)
    depth,_,_ = cv2.split(depth_dilate)
    return depth

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(SOURCE,BASE_NAME):
    view_img = False

    box_num = 0
    pred_coord = []
    img0 = SOURCE

    # Initialize
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(CFG, IMGSZ).cuda()
    model.load_state_dict(torch.load(WEIGHTS[0], map_location=device)['model'])
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Use single image as input of "detect(SOURCE)"
    img_size = IMGSZ
    auto_size = 64

    # Get names and colors
    names = load_classes(NAMES)
    # 변경2 
    flower_color = (76,0,153)    # Img 상에 결과 출력할때 위한 글씨 색, 이후 plot_one_box에서 사용
#                                    오이에는 필요 X
                                   
    branch_color = (255,128,0)   # Img 상에 결과 출력할때 위한 글씨 색
    # for tomato
    # 변경 3
    if N == 1:
        colors = [flower_color, branch_color]
    elif N == 2:
    # for cucumber
        colors = [branch_color]


    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, IMGSZ, IMGSZ), device=device)  # init img
    # print("Shape of img: ", img.shape)

    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    # Padded resize
    img = letterbox(img0, new_shape=img_size, auto_size=auto_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # t1 = time_synchronized()
    s = ''
    pred = model(img, augment=AUGMENT)[0]

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
    # print("pred: ", pred)
    # t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # normalization gain whwh
        
        save_path = "inference/output/" + BASE_NAME
        txt_path = "inference/output/" + Path(BASE_NAME).stem
        s += '%gx%g ' % img.shape[2:]  # print string

        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            # print("det :",det)
            for *xyxy, conf, cls in det:
                if cls > 10:
                    pass
                else:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                    # print("\n")
                    # print("xywh: ", xywh) #check out result of xywh
                    classNum2arr = np.array([int(cls)])
                    # print("classNum2arr: ", classNum2arr)
                    coord = np.array(xywh)
                    # print("box_num:", box_num)
                    
                    # print("\n cls:", cls)
                    pred_coord.append(np.concatenate((classNum2arr, coord)))

                    with open(txt_path + '.txt', 'a') as f: # Write to file
                        # f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                        f.write(('%g ' * 6 + '\n') % (cls, conf, *xywh))  # label format
                    # print("det : ", det)
                    # print("xywh :", xywh)
                    # print("conf :", conf)
                    # print("names:", names)
                    # print("cls :", cls)
                    label = '%s %.2f' % (names[int(cls)], conf) # Add bbox to image
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)

                    # Save results (image with detections)
                    cv2.imwrite(save_path, img0)

    return pred_coord


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness,cv2.LINE_AA)

    return text_size


# In[4]:



def inputProcessing(color_img, depth_img):

    # in real-time, the stored depth values = .abc (ex. 0.5), unit: m
    # codes below are written by data set in ABC(ex. 500), unit: mm
    # depth_img = depth_img*1000 # check the depth value unit and add this line to meet the scale used in this code

    model_type = MODEL_TYPE
    img_h = color_img.shape[0]
    img_w = color_img.shape[1]

    depth_img = morphology(depth_img)
    if model_type == "blur9":
        kernel = np.ones((9,9))/9**2
        result = color_img
        result = cv2.filter2D(result, -1, kernel) # -1 : 원본 이미지와 같은 depth

        for i in range(img_h):
            for j in range(img_w):
                px_depth = depth_img[i,j]
                px_rgb = color_img[i,j]                
                if px_depth != 0: 
                    result[i,j] = px_rgb
        blur9_img = result
        return blur9_img
        
    elif model_type == "blur21":
        kernel = np.ones((21,21))/21**2
        result = color_img
        result = cv2.filter2D(result, -1, kernel)

        for i in range(img_h):
            for j in range(img_w):
                px_depth = depth_img[i,j]
                px_rgb = color_img[i,j]            
                if px_depth != 0: 
                    result[i,j] = px_rgb
        blur21_img = result
        return blur21_img

    elif model_type == "blur100":
        kernel = np.ones((100,100))/100**2
        result = color_img
        result = cv2.filter2D(result, -1, kernel)

        for i in range(img_h):
            for j in range(img_w):
                px_depth = depth_img[i,j]
                px_rgb = color_img[i,j]
                if px_depth != 0: 
                    result[i,j] = px_rgb
        blur100_img = result
        return blur100_img

    elif model_type == "greyAbs":
        depth_shortRange = depth_img
        result = np.zeros((numH,numW,3))
        result[:,:,0] = depth_img

        for i in range(img_h):
            for j in range(img_w):
                px = depth_shortRange[i,j]
                # px = px*1000
                if px > 1000:
                    px = 0
                    depth_shortRange[i,j] = px

                px_1000to255 = depth_shortRange[i,j]
                px_1000to255 = int(px_1000to255*0.255)
                result[i,j,0] = px_1000to255      
        greyAbs_img = result
        return greyAbs_img

    elif model_type == "subtr":
        result = color_img
        depth_subtr = depth_img

        for i in range(img_h):
            for j in range(img_w):
                px = depth_subtr[i,j]
                # px = px*1000
                if px > 1000:
                    px = 0
                    depth_subtr[i,j] = px    
    
        for i in range(img_h):
            for j in range(img_w):
                px_depth = depth_subtr[i,j]
                px_rgb = color_img[i,j]
                if px_depth == 0: 
                    result[i,j] = 0
        subtr_img = result
        return subtr_img

    else:
        return color_img
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def cut_indices(numbers, thres):
    # this function iterate over the indices that need to be 'cut'
    for i in range(len(numbers)-1):
        if numbers[i+1] - numbers[i] > thres:
            yield i+1

def splitter(numbers, thres):
    # this function split the original list into sublists.
    px = 0
    for x in cut_indices(numbers, thres):
        yield numbers[px:x]
        px = x
    yield numbers[px:]

def cluster(numbers, thres):
    # using the above result, to form a dict object.
    cluster_ids = range(1,len(numbers))
    return dict(zip(cluster_ids, splitter(numbers, thres)))

def px2mm(px_value,depth_value,focalLen):
    mm_value = (px_value * depth_value)/focalLen
    return mm_value

def mm2px(mm_value,depth_value,focalLen):
    px_value = (mm_value * focalLen)/depth_value
    return px_value



error_cnt = 0
error_list = [0 for i in range(7)]
def getProcessing(color_img, depth_img):
    global error_cnt
    global error_list
    global target_folder
    inference_dir = "./inference/output"
    pp_dir = target_folder + "/preprocessing"
    
    model_type_dir = pp_dir + "/" + str(MODEL_TYPE)
    

    # create_folder(proc_dir)
    createFolder(inference_dir)
    
    createFolder(model_type_dir)

    createFolder(model_type_dir)
    # #
    # color_img = rgb.copy()
    # original_rgb = rgb.copy()
    # depth_img = depth.copy()

    # showme(input_img)

    input_img  = inputProcessing(color_img, depth_img)
    # cv2.imwrite 함수 이용해 pre_dir 폴더에 BASE_NAME 에 저장된 파일이름으로 inputProcessing된 image 저장
    cv2.imwrite(model_type_dir+"/"+str(MODEL_TYPE) + "_" + os.path.basename(BASE_NAME),input_img) # save processed image
    # showme(input_img)
    print("\n현재 입력된 이미지 : ", BASE_NAME)
    print("저장 경로 :",model_type_dir+"/"+str(MODEL_TYPE) + "_" + os.path.basename(BASE_NAME))
    print("\n")
    
    

# base_folder = 'test'

# def folder_list(dirname):
#     folders = []
#     for filename in os.listdir(dirname):
#         file_path = os.path.join(dirname, filename)
#         if os.path.isdir(file_path):            
#             folders.append(file_path)
#     return sorted(folders)
# folders = folder_list(base_folder)

# for i,name in enumerate(folders):
#     print("{} : {}".format(name, i))

# print("number of dir 입력 :")
# target_folder = folders[int(input())]

root = Tk()
root.title("Downloader")
root.geometry("540x300+100+100")
root.resizable(False, False)

root.dirName=filedialog.askdirectory()
#root.file = filedialog.askopenfile(initialdir='path', title='select file', filetypes=(('jpeg files', '*.jgp'), ('all files', '*.*')))
target_folder = str(root.dirName)

RGB_dir1 = target_folder + '/rgb/*.png'
RGB_dir2 = target_folder + '/rgb/*.jpg'
depth_dir = target_folder + '/depth/*.csv'
color_dir = target_folder + '/color_depth/*.png'
    

rgbrgb = int(input("png : 1, jpg : 2"))
if rgbrgb == 1:
    RGB_files = sorted(glob.glob(RGB_dir1)) # 디렉토리의 .png로 끝나는 모든 파일 저장
elif rgbrgb == 2:
    RGB_files = sorted(glob.glob(RGB_dir2)) # 디렉토리의 .jpg로 끝나는 모든 파일 저장

n = len(RGB_files)
# if RGB_files_jpg == [] and RGB_files_png == []:
#     print("rgb folder is emtpy")
# elif RGB_files_jpg == [] and RGB_files_png != []:
#     RGB_files = RGB_files_png
# elif RGB_files_jpg != [] and RGB_files_png == []:
#     RGB_files = RGB_files_jpg
# else:
#     RGB_files = [RGB_files_png, RGB_files_jpg]
    
# print("RGB :", RGB_files)
DEPTH_files = sorted(glob.glob(depth_dir)) # 디렉토리의 .csv 끝나는 모든 파일 저장
COLORD_files = sorted(glob.glob(color_dir)) # 디렉토리의 .png로 끝나는 모든 파일 저장
print("dir : ", target_folder)
cnt_total = 0.0001

for i in range(6,7):
    # m = int(input("MODEL_TYPE 입력 : \nsubtr = 1\nblur21 = 2\nblur100 = 3\n blur9 = 4\noriginal rgb = 5\ngreyAbs = 6\n"))
    m = i
    if m == 1:
        MODEL_TYPE = "subtr"
        print(MODEL_TYPE)
    elif m == 2:
        MODEL_TYPE = "blur21"
    elif m == 3:
        MODEL_TYPE = "blur100"
    elif m == 4:
        MODEL_TYPE = "blur9"
    elif m == 5:
        MODEL_TYPE = "rgb"
    elif m == 6:
        MODEL_TYPE = "greyAbs"

    for (rgb,depth) in zip(RGB_files,DEPTH_files):
        if cnt_total == 0.0001:
            pass
        else:
            print("현재 저장된 이미지 : ", )
            print("성공 : {} / {}".format(round(cnt_total - error_cnt) , round(n)))
        BASE_NAME = os.path.basename(rgb) # RGB_files에서 얻어온 rgb 파일의 파일이름만 반환
        # https://url.kr/nerowx
        BASE_NAME2 = Path(depth).stem+".png"
        # BASE_NAME3 = os.path.basename(color_depth)

        rgb= cv2.imread(rgb, cv2.IMREAD_COLOR)
        df = read_csv(depth, header = None, low_memory = False)

        depth = df.values
        depth = np.asanyarray(depth)
        ### 2022.04.29 추가
        for i in range(depth.shape[1]):
            depth[0,i] = int(0)
        ###
        
        depth_edit = depth
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                ele = float(depth[i,j])
                ele = round(ele*1000)
                depth[i,j] = ele
        
        # depth = depth_edit
        # color_depth = cv2.imread(color_depth, cv2.IMREAD_COLOR)
        
        original_rgb = rgb.copy()
        color_img = rgb
        depth_img = depth
        # color_depth = color_depth
        cnt_total += 1
        getProcessing(color_img, depth_img)
    print(cnt_total)
    cnt_total = 0
    
# # Error Plot
# X = ['E-Code1','E-Code2','E-Code3','E-Code4','E-Code5','E-Code6',
#      'E-Code7']
# plt.barh(X,error_list)
# plt.savefig("./vision_processing/error_report/error_report.png")

# print("")
# if n == 1:
#     print("Stem Diameter 검출을 위해 사용된 Main Value : Depth")
# elif n == 2:
#     print("Stem Diameter 검출을 위해 사용된 Main Value : RGB Mean Value by Depth")
# elif n == 3:
#     print("Stem Diameter 검출을 위해 사용된 Main Value : HSV Mean Value by Depth")
# print("")

# print("성공률 : {} %".format(round(100-error_cnt/cnt_total*100),3))

print("process finished")


