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



N = int(input("Tomato : 1, Cucumber : 2 \n"))
if N == 1:
    IMGSZ = 320
    WEIGHTS = ["weights/yolor/exp_aug/weights/best_overall.pt"]
    NAMES = "data/farmbot_tomato.names"
    MODEL_TYPE = "aug"
elif N == 2:
    IMGSZ = 448
    WEIGHTS = ["weights/yolor/yolor_p675/weights/best_overall.pt"]
    NAMES = "data/farmbot_cucumber.names"
    MODEL_TYPE = "rgb"
# 변경
   

# MODEL_TYPE = "subtr"
# MODEL_TYPE = "blur21"
# MODEL_TYPE = "blur100"
# MODEL_TYPE = "blur9"
# MODEL_TYPE = "rgb"
# MODEL_TYPE = "greyAbs"

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

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

# Initialize
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
######################################################################################
# Load model
model = Darknet(CFG, IMGSZ).cuda()
model.load_state_dict(torch.load(WEIGHTS[0], map_location=device)['model'])
model.to(device).eval()
if half:
    model.half()  # to FP16

def detect(SOURCE,BASE_NAME):
    view_img = False

    box_num = 0
    pred_coord = []
    img0 = SOURCE

    # # Initialize
    # device = select_device(DEVICE)
    # half = device.type != 'cpu'  # half precision only supported on CUDA

    # # Load model
    # model = Darknet(CFG, IMGSZ).cuda()
    # model.load_state_dict(torch.load(WEIGHTS[0], map_location=device)['model'])
    # model.to(device).eval()
    # if half:
    #     model.half()  # to FP16

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


# ### 오이

# # In[64]:


# base_folder = './test'

# def folder_list(dirname):
#         folders = []
#         for filename in os.listdir(dirname):
#             file_path = os.path.join(dirname, filename)
#             if os.path.isdir(file_path):            
#                 folders.append(file_path)
#         return sorted(folders)
# folders = folder_list(base_folder)

# for i,name in enumerate(folders):
#     print("{} : {}".format(name, i))

# print("number of dir 입력 :")
# target_folder = folders[int(input())]
    
# # 변경 : 토마토 sample
# if N == 1:
#     RGB_dir = target_folder + '/rgb/RGB_20220222_017_3.png'
#     depth_dir = target_folder + '/depth/dist_20220222_017_3.csv'
#     color_dir = target_folder + '/color_depth/dist_20220222_017_3.png'
# elif N == 2:
#     # 변경 : 오이 sample
#     RGB_dir = target_folder + '/rgb/RGB_20220426_035.png'
#     depth_dir = target_folder + '/depth/dist_20220426_035.csv'
#     color_dir = target_folder + '/color_depth/dist_20220426_035.png'

# # target_folder = folders[int(input())

# rgb= cv2.imread(RGB_dir, cv2.IMREAD_COLOR)
# df = read_csv(depth_dir, header = None, low_memory = False)

# depth = df.values
# depth = np.asanyarray(depth)

# ### 2022.04.29 추가
# for i in range(depth.shape[1]):
#     depth[0,i] = int(0)
# ###

# depth_edit = depth
# for i in range(depth.shape[0]):
#     for j in range(depth.shape[1]):
#         ele = float(depth[i,j])
#         ele = round(ele*1000)
#         depth[i,j] = ele

# # depth = depth_edit
# color_depth = cv2.imread(color_dir, cv2.IMREAD_COLOR)

# BASE_NAME = os.path.basename(RGB_dir)
# BASE_NAME2 = Path(depth_dir).stem+".png"
# BASE_NAME3 = os.path.basename(color_dir)

# # rgb, depth, color_depth

# print(BASE_NAME, BASE_NAME2, BASE_NAME3)


# ## 0509(월), 만든 detect

# In[48]:


error_cnt = 0
error_list = [0 for i in range(7)]
pred_csv = pd.DataFrame({'이름':[],'예측두께':[], '예측길이':[]})
def getProcessing(color_img, depth_img, color_depth):
    global error_cnt,error_list,pred_csv
    inference_dir = "./inference/output"
    proc_dir = "./vision_processing"

    growthPoint_dir = proc_dir + "/growthPoint"
    growthPoint_depth_dir = proc_dir + "/growthPoint_depth"

    result_dir = proc_dir + "/result"
    crop_dir = proc_dir + "/crops"
    edge_dir = proc_dir + "/edges"
    bin_dir = proc_dir + "/bins"

    whole_dir = proc_dir + "/whole_depth"
    erode_dir = proc_dir + "/erode"
    dil_dir = proc_dir + "/dil"
    pre_dir = proc_dir + "/pre"
    
    error_dir = proc_dir + "/error_report"

    # create_folder(proc_dir)
    createFolder(inference_dir)
    createFolder(growthPoint_dir)
    createFolder(growthPoint_depth_dir)
    createFolder(result_dir)

    createFolder(crop_dir)
    createFolder(edge_dir)
    createFolder(bin_dir)

    # createFolder(erode_dir)
    createFolder(dil_dir)
    createFolder(pre_dir)

    createFolder(whole_dir)
    
    createFolder(error_dir)

    # #
    # color_img = rgb.copy()
    # original_rgb = rgb.copy()
    # depth_img = depth.copy()

    # showme(input_img)

    input_img  = inputProcessing(color_img, depth_img)
    # cv2.imwrite 함수 이용해 pre_dir 폴더에 BASE_NAME 에 저장된 파일이름으로 inputProcessing된 image 저장
    cv2.imwrite(pre_dir+"/"+os.path.basename(BASE_NAME),input_img) # save processed image
    # showme(input_img)
    print("\n ---------------------------------")
    print("\n현재 입력된 이미지 : ", BASE_NAME)
    print("\n")
    # getProcessing 1. Depth Image 없을 때 Error Code = 1
    if depth_img is None:
        error_code = 1
        print("\n")
        print("[Error-Code1] There is no depth image! Please capture the input image again")
        print("\n")
        with open(result_dir+"/"+Path(os.path.basename(BASE_NAME)).stem + '.txt', 'a') as f: # Write predicted growth information to file
            f.write(('%g ' * 1 + '\n') % (error_code))  #result
        error_cnt += 1
        error_list[0] += 1
        return error_code

    else:
        # getProcessing 2. Depth 있을때 Depth img 처리
        # try 1
        # uint8 은 0~255
        # img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U) 이후에 시도해보기
        # 센서로 읽은 값 0~1000이므로 astype("uint8") 이용해서 0~255 사이 값으로 바꿔주기
        
        # horizontal_focalLen = 897.507 # w_depth, h_depth: 1280, 720
        horizontal_focalLen = 1300.507 # w_depth, h_depth: 1280, 720
        vertical_focalLen = 898.292
        
        img = depth_img.astype("uint8") # load depth image to process image with OpenCV 
        depth_values = depth_img.astype("uint16") # load depth value to tranform pixel distance to mm

        # initialize
        heights1 = [] # initialize the list variable "heights"
        heights2 = [] # initialize the list variable "heights"
        heights3 = [] # initialize the list variable "heights"

        depth_threshold = 0 # initialize the int variable "depth_threshold"
        plantHeight = 0

        # getProcessing 3. Color Image 없을 때 Error Code = 2
        if color_img is None:
            error_code = 2
            print("\n")
            print("[Error-Code2] There is no color image! Please capture the input image again")
            print("\n")
            with open(result_dir+"/"+Path(os.path.basename(BASE_NAME)).stem + '.txt', 'a') as f: # Write predicted growth information to file
                f.write(('%g ' * 1 + '\n') % (error_code))  #result
            error_cnt += 1
            error_list[1] += 1
            return error_code
        else:
            with torch.no_grad(): # Pytorch autograd engine 꺼서 자동으로 gradient 계산 X
                if UPDATE:  # update all models (to fix SourceChangeWarning)
                    for WEIGHTS in ['']:
                        coord = detect(input_img,BASE_NAME)
                        strip_optimizer(WEIGHTS)                
                else:
                    # detect 함수 사용해 YOLOr 통한 bbox coord 구함
                    coord = detect(input_img,BASE_NAME)
            # getProcessing 4. bbox 찾는데 실패해 좌표값을 얻지 못했을때 Error Code = 3
            
            ############## bbox 여러개일때 processing 1. depth로 걸러주기 start ##############
            centerX_list_picture_scale = []
            centerY_list_picture_scale = []

            for i in range(len(coord)):
                centerX_list_picture_scale.append(round(coord[i][1]*numW))
                centerY_list_picture_scale.append(round(coord[i][2]*numH))

            number_of_bboxes = len(coord)

            # for i in coord:
            #     print(i)
            cnt_coord = 0
            for i in range(number_of_bboxes):
                y = centerY_list_picture_scale[i]
                x = centerX_list_picture_scale[i]
                if depth_values[y][x] == 0:
                    # print("x :", x)
                    # print("y :", y)
                    # print("depth_values[y][x] :", depth_values[y][x])
                    del coord[i-cnt_coord]
                    cnt_coord += 1
            ############## bbox 여러개일때 processing 1. depth로 걸러주기 end ##############
            if coord == [] or len(coord) < 2: 
                error_code = 3
                print("\n")
                # print("[Error-Code3] No bounding boxes are detected! Model cannot find target objects in the input image. Please use another input image.")
                print("[Error-Code3] Detected bounding boxes are not enough! Model can find just few or none target objects in the input image. Please use another input image.")
                print("\n")
                error_cnt += 1
                error_list[2] += 1
                with open(result_dir+"/"+Path(os.path.basename(BASE_NAME)).stem + '.txt', 'a') as f: # Write predicted growth information to file
                    f.write(('%g ' * 1 + '\n') % (error_code))  #result
                return error_code
            
            
            
            else:
                # bbox도 찾았다면
                # coord 형태 
                # ex. 만약 꽃 3, 분기점 1 있다면
                # [array([0,0.46133,0.75417,0.058594,0.075]), 
                #  array([0,0.4082,0.81667,0.027344,0.055556]), 
                #  array([1,0.47344,0.85,0.042188,0.11667]), 
                #  array([0,0.4168,0.69444,0.030469,0.077778])]
                
                
                list_dist = []
                x1_list = []
                x2_list = []
                y1_list = []
                y2_list = []

                centerX_list = []
                centerY_list = []
                centerX_list_ = []
                centerY_list_ = []
                
                tot_list = []
                for idx,i in enumerate(coord):
                    centerX = i[1] #branch의 BBox 중앙 x좌표
                    centerY = i[2] #branch의 BBox 중앙 y좌표
                    boxW = i[3]    #branch의 BBox 가로 너비
                    boxH = i[4]    #branch의 BBox 세로 높이
                    half_boxW = boxW/2
                    half_boxH = boxH/2
                    
                    roi_centerX = round(centerX*numW)
                    roi_centerY = round(centerY*numH)
                    
                    # Bbox의 좌우,상하 끝 좌표 1280x720 맞게 변환
                    rectPoint_x1 = round((centerX-half_boxW)*numW)
                    rectPoint_x2 = round((centerX+half_boxW)*numW)
                    rectPoint_y1 = round((centerY-half_boxH)*numH)
                    rectPoint_y2 = round((centerY+half_boxH)*numH)
                    
            
                    # bbox 영역의 평균 depth값을 얻기 위한 crop
                    roi_img = img[rectPoint_y1:rectPoint_y2,rectPoint_x1:rectPoint_x2].copy()
                    roi_depth = depth_values.copy()
                    roi_depthCpy = roi_depth.copy()
                    
                    cropDpth = roi_depthCpy[roi_centerY-15:roi_centerY+15, roi_centerX-15:roi_centerX+15]
                    roi_depth_ = cropDpth[cropDpth>0]
                    roi_depth = np.nanmedian(roi_depth_)
                    # roi_depth2 = roi의 중앙점의 depth
                    
                    # Bbox의 중심점 좌표 1280x720 맞게 변환
                    centerX_ = round((rectPoint_x1+rectPoint_x2)/2)
                    centerY_ = round((rectPoint_y1+rectPoint_y2)/2)
                    
                    # list에 추가
                    x1_list.append(rectPoint_x1)
                    x2_list.append(rectPoint_x2)
                    y1_list.append(rectPoint_y1)
                    y2_list.append(rectPoint_y2)

                    # 변환된 center 좌표 list
                    centerX_list_.append(centerX_)
                    centerY_list_.append(centerY_)

                    # original center 좌표 list
                    centerX_list.append(centerX)
                    centerY_list.append(centerY)
                    
                    centerX_candid = centerX_list_[idx]
                    centerY_candid = centerY_list_[idx]

                    ##?? 
                    # x,y 거리정보 : |변환된 좌표 - 이미지 size / 2|
                    x_dist = abs(centerX_candid-(numW/2))
                    y_dist = abs(centerY_candid-(numH/2))

                    tot_dist = x_dist**2 + y_dist**2

                    tot_list.append(tot_dist)
                    
                ############## bbox 여러개일때 processing 2. distance로 걸러주기 start ##############
                    
                ############## dist 1
                
                tot_list_sorted = sorted(tot_list)
                
                fin_indx1 = tot_list.index(tot_list_sorted[0])
                fin_indx2 = tot_list.index(tot_list_sorted[1])
            
                # 거리 가장 짧은 bbox의 center 좌표 in total image scale
                centerX1 = round(centerX_list[fin_indx1]*numW)
                centerY1 = round(centerY_list[fin_indx1]*numH)
                
    #             # 두 번째 거리 짧은 bbox 검출 위해 가장 짧은 bbox pop
    #             tot_list.pop(fin_indx1)
    #             centerX_list.pop(fin_indx1)
    #             centerY_list.pop(fin_indx1)
                
                
                ############## dist 2
                
    #             fin_indx2 = fin_indx1 + 1
                
                # 거리 가장 짧은 bbox의 center 좌표 in total image scale
                centerX2 = round(centerX_list[fin_indx2]*numW)
                centerY2 = round(centerY_list[fin_indx2]*numH)
                
                ############## bbox 여러개일때 processing 2. distance로 걸러주기 end ##############
                
                ############## depth dist 구하기
                d1 = int(depth_values[centerY1][centerX1])
                d2 = int(depth_values[centerY2][centerX2])
                distance_depth = abs(d1-d2)
                
                ############## Gradient 통해 Error 처리 start ##############
                
                gradient = abs(centerY2 - centerY1) / abs(centerX2 - centerX1 + eps)
                angle_of_gradient = 90 - np.arctan(gradient)*180/pi
                # print("현재 detect된 branch point 사이의 직선의 기울기 :", angle_of_gradient)
                if angle_of_gradient > 30: #각도가 30도보다 클 때
                    error_code = 4
                    print("\n")
                    print("[Error-Code4] We dected bounding boxes from two seperated cucumbers")
                    print("\n")
                    error_cnt += 1
                    error_list[3] += 1
                    with open(result_dir+"/"+Path(os.path.basename(BASE_NAME)).stem + '.txt', 'a') as f: # Write predicted growth information to file
                                    f.write(('%g ' * 1 + '\n') % (error_code))  #result
                    return error_code
                
                ############## Gradient 통해 Error 처리 end ##############
                    
                else:   
                    ############## Branch to Branch 거리구하기
                    dist_between_bboxes_px = sqrt((centerX1 - centerX2) ** 2 + 
                                            (centerY1 - centerY2) ** 2)
                    dist_between_bboxes_mm = px2mm(dist_between_bboxes_px, roi_depth, vertical_focalLen)
                    dist_between_bboxes_mm = sqrt(dist_between_bboxes_mm ** 2 + distance_depth ** 2)
        #             print("distance between in pixel : %10.3f"%dist_between_bboxes_px)
                    # print("distance between in mm : %10.3f"%dist_between_bboxes_mm)
                    
                    ############## 줄기 두께 구하기
        #             1. 두 개의 bbox 중 y좌표가 더 큰(더 아래에 있는) bbox 선정
                    if centerY1 > centerY2:
                        fin_indx = fin_indx1
                    elif centerY1 < centerY2:
                        fin_indx = fin_indx2
                        
                    # 2. 기존 알고리즘 적용
                    rectPoint_x1 =  x1_list[fin_indx]
                    rectPoint_x2 =  x2_list[fin_indx]
                    rectPoint_y1 =  y1_list[fin_indx]
                    rectPoint_y2 =  y2_list[fin_indx]

                    centerX = centerX_list[fin_indx]
                    centerY = centerY_list[fin_indx]

                    branch_centerX = round(centerX*numW)
                    branch_centerY = round(centerY*numH)

                    branch_img = img[rectPoint_y1:rectPoint_y2,rectPoint_x1:rectPoint_x2].copy()
                    branch_depth = depth_values.copy()
                    branch_depthCpy = branch_depth.copy()
                    ## 삭제
                    # showme(branch_img)
                    # +- 15 범위로 crop
                    cropDpth = branch_depthCpy[branch_centerY-15:branch_centerY+15,branch_centerX-15:branch_centerX+15]
                    roi_depth_ = cropDpth[cropDpth>0]
                    roi_depth = np.nanmedian(roi_depth_) # depth value used as a standard: type as uint16
                    #####################bounding box가 너무 작을때####################
                    
                    MM_TH_DISTANCE = 40
                    PX_TH_DISTANCE = mm2px(MM_TH_DISTANCE,roi_depth,vertical_focalLen)

                    gap_Y = rectPoint_y2-rectPoint_y1
                    gap_X = rectPoint_x2-rectPoint_x1

                    if gap_Y < PX_TH_DISTANCE:

                        r = 0.5

                        rectPoint_y1 = round(rectPoint_y1 - gap_Y * r)
                        rectPoint_y2 = round(rectPoint_y2 + gap_Y * r)
                        rectPoint_x1 = round(rectPoint_x1 - gap_X * r)
                        rectPoint_x2 = round(rectPoint_x2 + gap_X * r)
                        
                        branch_img = img[rectPoint_y1:rectPoint_y2,rectPoint_x1:rectPoint_x2].copy()
                    
                ################################################################
                    # Move the coordinates of lowest side of the bounding box to 0
                    boundingBoxH = rectPoint_y2-rectPoint_y1
                    centerH = round(boundingBoxH/2) # 왜 있는 line?

                    MM_DISTANCE = 20 # MM_DISTANCE가 뭐지? #?
                    PIXEL_DISTANCE = mm2px(MM_DISTANCE,roi_depth,vertical_focalLen)

                    # crop image of ROI
                    # 앞서 지정한 low_rate(논문에서는 r_lower), high_rate(논문에2서는 r_upper)에 따라 ROI crop하기
                    min_ROIh = round(branch_img.shape[0]*low_rate)
                    max_ROIh = round(branch_img.shape[0]*high_rate)
                    min_ROIw = round(branch_img.shape[1]*0)
                    max_ROIw = round(branch_img.shape[1]*1)


                    Ycoord = (min_ROIh+max_ROIh)/2 - PIXEL_DISTANCE
                    # print("현재 입력된 이미지 : ", BASE_NAME)
                    Ycoord = round(Ycoord) # Ycoord: pixel at center point
                    # print("ok 1")
                    while Ycoord < 3:
                        # 만약 Ycoord가 3보다 작다면 MM_DISTANCE를 1씩 줄여가며 Ycoord를 다시 구해준다.
                        MM_DISTANCE = MM_DISTANCE - 1

                        # PIXEL_DISTANCE = mm2px(MM_DISTANCE,depth_value,vertical_focalLen)
                        PIXEL_DISTANCE = mm2px(MM_DISTANCE,roi_depth,vertical_focalLen)

                        Ycoord = (min_ROIh+max_ROIh)/2 - PIXEL_DISTANCE

                        Ycoord = round(Ycoord) # Ycoord: pixel at center point

                        if MM_DISTANCE == 0:
                            error_code = 5
                            print("\n")
                            print("[Error-Code5] Standard distance is larger than the whole height of detected bounding box of branch point.")
                            print("\n")
                            error_cnt += 1
                            error_list[4] += 1
                            with open(result_dir+"/"+Path(os.path.basename(BASE_NAME)).stem + '.txt', 'a') as f: # Write predicted growth information to file
                                f.write(('%g ' * 1 + '\n') % (error_code))  #result
                            return error_code

                        else:
                            continue

                    # print("MM_DISTANCE becomes: ", MM_DISTANCE)
                    branch_imgCpy = branch_img.copy()
                    ########################### COLOR ##############################

                    # color img crop
                    c_img = original_rgb[rectPoint_y1:rectPoint_y2,rectPoint_x1:rectPoint_x2].copy()
                    c_img = cv2.normalize(c_img, None, 0,255, cv2.NORM_MINMAX)
                    PADDING_DISTANCE1 = 13
                    PADDING_DISTANCE2 = 13
                    # bin img crop
                    ## 삭제
                    bin_crop = img[rectPoint_y1:rectPoint_y2,rectPoint_x1:rectPoint_x2].copy()
                    # showme(bin_crop)
                    ##################################################
                    for i in range(bin_crop.shape[0]):
                        for j in range(bin_crop.shape[1]):
                            pixel_value = bin_crop[i,j]
                            roi_depth_uint16 = roi_depth.astype("uint8")
                            if pixel_value>(roi_depth_uint16-PADDING_DISTANCE1) and pixel_value<(roi_depth_uint16+PADDING_DISTANCE2):
                                bin_crop[i,j] = 255
                            else:
                                bin_crop[i,j] = 0
                    ###################################################
                    # color와 bin 비교해가면서 pixel 값 뽑야내기
                    def subtract_bin_from_color(bin_crop, color_crop):
                        save_color = np.zeros(shape=(color_crop.shape[0],color_crop.shape[1],color_crop.shape[2]), 
                                                                                                dtype = np.uint8)
                        for k in range(color_crop.shape[2]):
                            for i in range(color_crop.shape[0]):
                                for j in range(color_crop.shape[1]):
                                    if bin_crop[i,j] == 255:
                                        save_color[i,j,k] = color_crop[i,j,k]
                                    else:
                                        save_color[i,j,k] = 0
                        return save_color
                    save_color_ = subtract_bin_from_color(bin_crop, c_img)

                    # depth로 뽑아낸 color 영역의 pixel 중앙값 
                    roi_colorB = np.nanmedian(save_color_[:,:,0][save_color_[:,:,0]>0])
                    roi_colorG = np.nanmedian(save_color_[:,:,1][save_color_[:,:,1]>0])
                    roi_colorR = np.nanmedian(save_color_[:,:,2][save_color_[:,:,2]>0])
                    # print(roi_colorB, roi_colorG, roi_colorR)

                    ### 2 Threshold 와 평균 RGB 값 이용한 추출

                    def subtract_color_by_mean(mean_B, mean_G, mean_R, color_img):
                        r = 1
                        thresholdB = np.std(color_img[:,:,0])*r
                        thresholdG = np.std(color_img[:,:,1])*r
                        thresholdR = np.std(color_img[:,:,2])*r
                        threshold_color_img = np.zeros(shape=(color_img.shape[0],color_img.shape[1],color_img.shape[2]),
                                                    dtype = np.uint8)

                        for i in range(color_img.shape[0]):
                            for j in range(color_img.shape[1]):
                                pixel_b = color_img[i,j,0]
                                pixel_g = color_img[i,j,1]
                                pixel_r = color_img[i,j,2]
                                if pixel_b > (mean_B-thresholdB) and pixel_b < (mean_B+thresholdB):
                                    if pixel_g > (mean_G-thresholdG) and pixel_g < (mean_G+thresholdG):
                                        if pixel_r > (mean_R-thresholdR) and pixel_r < (mean_R+thresholdR):
                                            threshold_color_img[i,j,0] = pixel_b
                                            threshold_color_img[i,j,1] = pixel_g
                                            threshold_color_img[i,j,2] = pixel_r
                                        else:
                                            threshold_color_img[i,j,0] = 0
                                            threshold_color_img[i,j,1] = 0
                                            threshold_color_img[i,j,2] = 0
                        return threshold_color_img

                    thres_img = subtract_color_by_mean(roi_colorB, roi_colorG, roi_colorR, c_img)

                    ### 3 중간 점검 - Save and Show
                    # print("ok 2")
                    # 저장하고 띄워주는 section
                    my_dir_color = proc_dir + "/1.color" + "/"+"".join(list(os.path.basename(BASE_NAME))[:-4])
                    createFolder(my_dir_color)
                    cv2.imwrite(my_dir_color+"/1"+os.path.basename(BASE_NAME), bin_crop)
                    cv2.imwrite(my_dir_color+"/2"+os.path.basename(BASE_NAME), c_img)
                    cv2.imwrite(my_dir_color+"/3"+os.path.basename(BASE_NAME), save_color_)
                    cv2.imwrite(my_dir_color+"/4"+os.path.basename(BASE_NAME), thres_img)


                    # showme(bin_crop)
                    # showme(c_img)
                    # showme(save_color_)
                    # showme(thres_img)

                    # print(my_dir_color)
                    # print(proc_dir + "/1.color")

                    ### 4 opencv 함수 이용한 binarization 

                    # binarization
                    imgray = cv2.cvtColor(thres_img, cv2.COLOR_BGR2GRAY)
                    ret,c2b_img_cv2 = cv2.threshold(imgray,10,255,cv2.THRESH_BINARY)

                    # showme(c2b_img)
                    cv2.imwrite(my_dir_color+"/5"+os.path.basename(BASE_NAME), c2b_img_cv2)
                    # showme(c2b_img_cv2)

                    ### 5 close 연산 (dilate and erode)

                    def close(target_bin_img):
                        c_img = 255 - target_bin_img
                        kernel = np.ones((3,3), np.uint8)
                        dil_ero = cv2.dilate(c_img,kernel, iterations = 1)
                        dil_ero = cv2.erode(dil_ero, kernel, iterations = 1)
                        dil_ero = 255 - dil_ero
                        return dil_ero
                    color_close_img = close(c2b_img_cv2)
                    cv2.imwrite(my_dir_color+"/6"+os.path.basename(BASE_NAME), color_close_img)
                    # showme(close_img)

                    ######################## New : HSV ###########################

                    def subtract_hsv_by_mean(th, mean_H, mean_S, mean_V, hsv_img):
                        threshold = th
                        threshold_hsv_img = np.zeros(shape=(hsv_img.shape[0],hsv_img.shape[1],hsv_img.shape[2]),
                                                    dtype = np.uint8)

                        for i in range(hsv_img.shape[0]):
                            for j in range(hsv_img.shape[1]):
                                pixel_h = hsv_img[i,j,0]
                                pixel_s = hsv_img[i,j,1]
                                pixel_v = hsv_img[i,j,2]
                                if pixel_h > (mean_H-threshold) and pixel_h < (mean_H+threshold):
                                    if pixel_s > (mean_S-threshold) and pixel_s < (mean_S+threshold):
                                        if pixel_v > (mean_V-threshold) and pixel_v < (mean_V+threshold):
                                            threshold_hsv_img[i,j,0] = pixel_h
                                            threshold_hsv_img[i,j,1] = pixel_s
                                            threshold_hsv_img[i,j,2] = pixel_v
                                        else:
                                            threshold_hsv_img[i,j,0] = 0
                                            threshold_hsv_img[i,j,1] = 0
                                            threshold_hsv_img[i,j,2] = 0
                        return threshold_hsv_img


                    my_hsv_dir = proc_dir + "/2.hsv"+ "/"+"".join(list(os.path.basename(BASE_NAME))[:-4])
                    compare_dir = proc_dir + "/3.compare"
                    createFolder(my_hsv_dir)
                    createFolder(compare_dir)
                    # color img crop
                    h_img = original_rgb[rectPoint_y1:rectPoint_y2,rectPoint_x1:rectPoint_x2].copy()
                    hsv_img = cv2.cvtColor(h_img, cv2.COLOR_BGR2HSV)
                    # bin_crop 그대로 이용 for depth 
                    # showme(h_img)
                    # showme(hsv_img)
                    h,s,v = cv2.split(hsv_img)
                    cv2.imwrite(my_hsv_dir+"/h.png", h)
                    cv2.imwrite(my_hsv_dir+"/s.png", s)
                    cv2.imwrite(my_hsv_dir+"/v.png", v)
                    # showme(h)
                    # showme(s)
                    # showme(v)

                    hsv_save_color = subtract_bin_from_color(bin_crop, hsv_img)
                    roi_colorH = np.nanmedian(hsv_save_color[:,:,0][hsv_save_color[:,:,0]>0])
                    roi_colorS = np.nanmedian(hsv_save_color[:,:,1][hsv_save_color[:,:,1]>0])
                    roi_colorV = np.nanmedian(hsv_save_color[:,:,2][hsv_save_color[:,:,2]>0])
                    # print(roi_colorH, roi_colorS, roi_colorV)

                    hsv_thres_img = subtract_hsv_by_mean(30,roi_colorH, roi_colorS, roi_colorV, hsv_img)
                    h_color_img = cv2.cvtColor(hsv_thres_img, cv2.COLOR_HSV2BGR)
                    # showme(h_color_img)




                    # binarization
                    hsv_imgray = cv2.cvtColor(h_color_img, cv2.COLOR_RGB2GRAY)
                    ret,c2b_hsv_cv2 = cv2.threshold(hsv_imgray,30,255,cv2.THRESH_BINARY)

                    # showme(c2b_img)
                    # showme(c2b_hsv_cv2)
                    hsv_close_img = close(c2b_hsv_cv2)


                    # showme(compare)
                    font = cv2.FONT_HERSHEY_PLAIN
                    numw,numh = hsv_close_img.shape
                    category = np.zeros(shape=(numw,numh,3),dtype = np.uint8)
                    categoryRGB = category.copy()
                    categoryHSV = category.copy()
                    categoryRGB = cv2.putText(categoryRGB, "RGB", (25,50),font,1,(255,255,255),1,cv2.LINE_AA)
                    categoryHSV = cv2.putText(categoryHSV, "HSV", (25,50),font,1,(255,255,255),1,cv2.LINE_AA)


                    add_bin = bin_crop.copy()
                    add_bin = cv2.cvtColor(add_bin, cv2.COLOR_GRAY2BGR)

                    add_Color_Depth = cv2.addWeighted(c_img, 1.2, add_bin,0.2 , 0)

                    compare = cv2.cvtColor(bin_crop, cv2.COLOR_GRAY2BGR)
                    compare = np.hstack([c_img, compare])
                    compare = np.hstack([compare, add_Color_Depth])

                    compare1 = np.hstack([c2b_img_cv2, color_close_img])
                    compare1 = cv2.cvtColor(compare1, cv2.COLOR_GRAY2BGR)
                    compare1 = np.hstack([compare, compare1])

                    # compare1 = np.hstack([bin_crop, c2b_img_cv2])
                    # compare1 = np.hstack([compare1, color_close_img])
                    # compare1 = cv2.cvtColor(compare1, cv2.COLOR_GRAY2BGR)
                    # compare1 = np.hstack([add_Color_Depth, compare1])
                    # compare1 = np.hstack([c_img, compare1])
                    compare1 = np.hstack([categoryRGB, compare1])

                    compare2 = np.hstack([c2b_hsv_cv2, hsv_close_img])
                    compare2 = cv2.cvtColor(compare2, cv2.COLOR_GRAY2BGR)
                    compare2 = np.hstack([compare, compare2])
                    # compare2 = np.hstack([bin_crop, c2b_hsv_cv2])
                    # compare2 = np.hstack([compare2, hsv_close_img])
                    # compare2 = cv2.cvtColor(compare2, cv2.COLOR_GRAY2BGR)
                    # compare2 = np.hstack([add_Color_Depth, compare2])
                    # compare2 = np.hstack([c_img, compare2])
                    compare2 = np.hstack([categoryHSV, compare2])
                    # showme(compare1)
                    # showme(compare2)
                    compare = np.vstack([compare1, compare2])




                    # print("ok 3")
                    # 저장하고 띄워주는 section
                    cv2.imwrite(my_hsv_dir+"/1"+os.path.basename(BASE_NAME), bin_crop)
                    cv2.imwrite(my_hsv_dir+"/2"+os.path.basename(BASE_NAME), h_img)
                    cv2.imwrite(my_hsv_dir+"/3"+os.path.basename(BASE_NAME), hsv_save_color)
                    cv2.imwrite(my_hsv_dir+"/4"+os.path.basename(BASE_NAME), h_color_img)
                    cv2.imwrite(my_hsv_dir+"/5"+os.path.basename(BASE_NAME), c2b_hsv_cv2)
                    cv2.imwrite(compare_dir+"/"+os.path.basename(BASE_NAME), compare)
                    # showme(h_img)
                    # showme(hsv_img)
                    # showme(h)
                    # showme(s)
                    # showme(v)
                    # showme(bin_crop)
                    # showme(hsv_save_color)
                    # showme(h_color_img)

                    #####################################################################

                    def depthorrgborhsv():
                        global cnt
                        global n
                        # cnt = 0 # 매 image마다 method 선택하려면 각주 해제
                        if cnt != 0:
                            if n == 1:
                                cropDst = bin_crop[min_ROIh:max_ROIh, min_ROIw:max_ROIw].copy()
                                return cropDst
                            elif n == 2:
                                cropDst = color_close_img[min_ROIh:max_ROIh, min_ROIw:max_ROIw].copy()
                                return cropDst
                            elif n == 3:
                                cropDst = hsv_close_img[min_ROIh:max_ROIh, min_ROIw:max_ROIw].copy()
                                return cropDst
                            else:
                                print("\n입력값이 잘못되었습니다. [1,2,3] 중 선택해주세요 \n")
                                depthorrgborhsv()

                        else:
                            print("\nDepth 이용 [1] RGB 이용한 mean[2] or HSV 이용한 mean?[3] :")
                            n = int(input())
                            if n == 1:
                                cropDst = bin_crop[min_ROIh:max_ROIh, min_ROIw:max_ROIw].copy()
                                cnt += 1
                                return cropDst
                            elif n == 2:
                                cropDst = color_close_img[min_ROIh:max_ROIh, min_ROIw:max_ROIw].copy()
                                cnt += 1
                                return cropDst
                            elif n == 3:
                                cropDst = hsv_close_img[min_ROIh:max_ROIh, min_ROIw:max_ROIw].copy()
                                cnt += 1
                                return cropDst
                            else:
                                print("\n입력값이 잘못되었습니다. [1,2,3] 중 선택해주세요 \n")
                                depthorrgborhsv()
                    cropDst = depthorrgborhsv()
                    # showme(bin_crop[min_ROIh:max_ROIh, min_ROIw:max_ROIw])

                    # cropDst = color_close_img[min_ROIh:max_ROIh, min_ROIw:max_ROIw].copy()

                    mean_bin_img = cropDst.copy()
                    cv2.imwrite(crop_dir+"/"+os.path.basename(BASE_NAME2),mean_bin_img) # save processed image

                    edges = cv2.Canny(mean_bin_img,700,750)
                    cv2.imwrite(edge_dir+"/"+os.path.basename(BASE_NAME2), edges)

                    img_output = cv2.cvtColor(mean_bin_img, cv2.COLOR_GRAY2BGR)

                    mmThickness_ofStem = 0
                    width = mean_bin_img.shape[1]   #.shape[1] : 열 개수 반환, 열 개수 for 너비 
                    height = mean_bin_img.shape[0]  #.shape[0] : 행 개수 반환, 행 개수 for 높이

                    stem_candidates1 = []
                    stem_candidates2 = []
                    stem_candidates3 = []

                    ## B-4) Measuring stem diameter of the target plant
                    Ycoord1 = Ycoord # choose 3 sample lines to decide more accurate thickness of stem
                    Ycoord2 = Ycoord-1
                    Ycoord3 = Ycoord-2
                    # print("ok 4")
                    # 가로 너비에 대해서 (line의 가로 픽셀 모두 비교하기 위함)
                    for j in range(width): # 오타(778 각주 whoose)
                        pixel1 = mean_bin_img[Ycoord1,j] # pixel: a variable to store the pixel value of a point whoose y-coordinate is the same with Ycoord
                        pixel2 = mean_bin_img[Ycoord2,j]
                        pixel3 = mean_bin_img[Ycoord3,j]

                        # 0~1 사이 값으로 normalize, 왜#?
                        norm_pixel1 = pixel1/255
                        norm_pixel2 = pixel2/255
                        norm_pixel3 = pixel3/255

                        # 만약 pixel 값이 1이라면(255인 부분), 줄기 후보 list에 추가 j 추가
                        if norm_pixel1==1:
                            stem_candidates1.append(j)

                        if norm_pixel2==1:
                            stem_candidates2.append(j)

                        if norm_pixel3==1:
                            stem_candidates3.append(j)

                    # j 번째 요소들이 추가된 줄기 후보 리스트의 길이 반환
                    length_stem1 = len(stem_candidates1)
                    length_stem2 = len(stem_candidates2)
                    length_stem3 = len(stem_candidates3)

                    length_stems = [length_stem1, length_stem2, length_stem3]
                    pixelDist_stem = np.nanmedian(length_stems) # 세 length 값 중 중앙값 반환
                    # print("ok 5")
                    ##?? 왜 그냥 중앙값을 length_stem으로 선택??
                    if pixelDist_stem == length_stem1:
                        stem_picked = stem_candidates1
                        YcoordPicked = Ycoord1

                    elif pixelDist_stem == length_stem2:
                        stem_picked = stem_candidates2
                        YcoordPicked = Ycoord2

                    elif pixelDist_stem == length_stem3:
                        stem_picked = stem_candidates3
                        YcoordPicked = Ycoord3
                    # print("ok 6")
                    # print(length_stems)
                    if len(stem_picked) != 0:
                        # print("ok 7")
                        # clustering 
                        stem_dict = cluster(stem_picked, 3)
                        stem_keyLst= list(stem_dict.keys())
                        cluster_lengthLst = []
                        stem_clusterLst = []

                        if len(stem_keyLst) == 0:
                            error_code = 6
                            error_cnt += 1
                            error_list[5] += 1
                            print("\n")
                            print("[Error-Code6] Clustered pixels of stem do not exist! Please use another input image.")
                            print("\n")
                            with open(result_dir+"/"+Path(os.path.basename(BASE_NAME)).stem + '.txt', 'a') as f: # Write predicted growth information to file
                                f.write(('%g ' * 1 + '\n') % (error_code))  #result
                            return error_code
                        else:
                            # print("ok 8")
                            for key in stem_keyLst:
                                stem_cluster = stem_dict[key]
                                cluster_length = len(stem_cluster)
                                cluster_lengthLst.append(cluster_length)
                                stem_clusterLst.append(stem_cluster)
                            # cluster 된 pixels 그룹 중 가장 긴 그룹 선택
                            stem_cluster_picked = cluster_lengthLst.index(max(cluster_lengthLst)) 
                            stem_picked=  stem_clusterLst[stem_cluster_picked]

                            # print("Current image: ", os.path.basename(BASE_NAME))
                            # print("stem_clusterLst: ", stem_clusterLst)
                            # print("stem_picked: ", stem_picked)
                            # 최종 원하는 결과값 : 줄기 두께(stem diameter)
                            pixelDist_stem = len(stem_picked)

                            startPicked = stem_picked[0]
                            endPicked = stem_picked[-1]

                            color_bin = cv2.cvtColor(mean_bin_img, cv2.COLOR_GRAY2BGR)

                            cv2.line(color_bin,(startPicked,YcoordPicked),(endPicked,YcoordPicked), stemROILine_color, 2)
                            cv2.imwrite(bin_dir+"/"+os.path.basename(BASE_NAME2),color_bin) # save processed image of ROI bin

                            kernel_stem = np.ones((3,3), np.uint8)
                            # 507 : img = depth_img.astype("uint8") # load depth image to process image with OpenCV 
                            whole_depth = img.copy()
                            whole_depth = 255 - whole_depth
                            whole_depth = cv2.morphologyEx(whole_depth, cv2.MORPH_OPEN, kernel_stem)
                            whole_depth = cv2.morphologyEx(whole_depth, cv2.MORPH_CLOSE,kernel_stem)
                            whole_depth  = cv2.medianBlur(whole_depth,3)
                            whole_depth = 255 - whole_depth

                            cv2.imwrite(dil_dir+"/"+os.path.basename(BASE_NAME2),whole_depth) # save processed image
                            
                            mmThickness_ofStem = px2mm(pixelDist_stem,roi_depth,horizontal_focalLen)
                            # print(branch_existence)
                            ############## Plot
                            if mmThickness_ofStem <= 2: # 5에서 2로 변경
                                error_code = 7
                                error_cnt += 1
                                error_list[6] += 1
                                print("\n")
                                print("[Error-Code7] Estimated thickness of stem is not reliable! ({}mm) Please use another input image.".format(round(mmThickness_ofStem,2)))
                                print("\n")
                                with open(result_dir+"/"+Path(os.path.basename(BASE_NAME)).stem + '.txt', 'a') as f: # Write predicted growth information to file
                                    f.write(('%g ' * 1 + '\n') % (error_code))  #result
                                return error_code
                            else:
        #                         if branch_existence!=0:
                                # showme(input_img)
                                result_img = input_img.copy()
                                standard = int(pixelDist_stem)
                                stem_Y = branch_centerY - (max_ROIh - (YcoordPicked + min_ROIh))

                                # text
                                
        #                             verticalLine_color = (0,255,204) # colors of lines on the result images
                                B2BLine_color = (0,255,204)

                                textThicknessOfStem = "Stem Diameter: "+str(round(mmThickness_ofStem,3))+"mm"
                                textB2B = "Upper Branch Point to Lower Branch Point : " + str(round(dist_between_bboxes_mm,3))  + "mm"
                                draw_text(result_img, textB2B,text_color = B2BLine_color, 
                                        pos=(numW-1050,numH-650),font=cv2.FONT_HERSHEY_SIMPLEX)
                                draw_text(result_img,textThicknessOfStem, text_color=stemROILine_color,
                                        pos=(numW-1050,numH-700),font=cv2.FONT_HERSHEY_SIMPLEX)
                                # 중심 사이 직선
                                cv2.line(result_img, (centerX1, centerY1),(centerX2, centerY2),(0,255,255), 2)

                                # bbox의 중심
                                cv2.circle(result_img, (centerX1, centerY1),2, (255,0,255),2)
                                cv2.circle(result_img, (centerX2, centerY2),2, (255,0,255),2)
                                cv2.line(result_img,(rectPoint_x1,stem_Y),(rectPoint_x2,stem_Y), stemLine_color, 2)
                                cv2.line(result_img,((rectPoint_x1+startPicked),stem_Y),(rectPoint_x1+startPicked+standard,stem_Y), stemROILine_color, 2)

                                cv2.imwrite(growthPoint_dir+"/"+os.path.basename(BASE_NAME), result_img)

                                tmp_df = pd.DataFrame(data=[[os.path.basename(BASE_NAME).split("_")[-1], mmThickness_ofStem, dist_between_bboxes_mm]],columns = ['이름', '예측두께', '예측길이'])
                                pred_csv = pd.concat([pred_csv,tmp_df])
    #                         showme(result_img)

base_folder = 'test'

def folder_list(dirname):
    folders = []
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isdir(file_path):            
            folders.append(file_path)
    return sorted(folders)
folders = folder_list(base_folder)

for i,name in enumerate(folders):
    print("{} : {}".format(name, i))

print("number of dir 입력 :")
target_folder = folders[int(input())]
RGB_dir1 = target_folder + '/rgb/*.png'
# RGB_dir2 = target_folder + '/rgb/*.jpg'
depth_dir = target_folder + '/depth/*.csv'
color_dir = target_folder + '/color_depth/*.png'
    


RGB_files = sorted(glob.glob(RGB_dir1)) # 디렉토리의 .jpg로 끝나는 모든 파일 저장
# RGB_files = sorted(glob.glob(RGB_dir2)) # 디렉토리의 .png로 끝나는 모든 파일 저장
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
for (rgb,depth,color_depth) in zip(RGB_files,DEPTH_files,COLORD_files):
    if cnt_total == 0.0001:
        pass
    else:
        print("성공 : {} / {}".format(round(cnt_total - error_cnt) , round(cnt_total)))
    BASE_NAME = os.path.basename(rgb) # RGB_files에서 얻어온 rgb 파일의 파일이름만 반환
    # https://url.kr/nerowx
    BASE_NAME2 = Path(depth).stem+".png"
    BASE_NAME3 = os.path.basename(color_depth)

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
    color_depth = cv2.imread(color_depth, cv2.IMREAD_COLOR)
    
    original_rgb = rgb.copy()
    color_img = rgb
    depth_img = depth
    color_depth = color_depth
    cnt_total += 1
    getProcessing(color_img, depth_img, color_depth)
    


print("")
if n == 1:
    print("Stem Diameter 검출을 위해 사용된 Main Value : Depth")
elif n == 2:
    print("Stem Diameter 검출을 위해 사용된 Main Value : RGB Mean Value by Depth")
elif n == 3:
    print("Stem Diameter 검출을 위해 사용된 Main Value : HSV Mean Value by Depth")
print("")

# X = ['E-Code1','E-Code2','E-Code3','E-Code4','E-Code5','E-Code6',
#      'E-Code7']
# X.reverse()
# error_list.reverse()
# plt.grid(True, axis = 'x',linestyle = ':', linewidth = 1)
# plt.xticks(range((max(error_list))+1))
# plt.barh(X,error_list)
X = ['E-Code1','E-Code2','E-Code3','E-Code4','E-Code5']
X.reverse()
er = error_list[2:6]
er.reverse()
er += [1]
plt.grid(True, axis = 'x',linestyle = ':', linewidth = 1)
plt.xticks(range((max(er))+1))
plt.barh(X,er)

plt.title("Error Code Distribution")
plt.xlabel('Error count')
plt.savefig("./vision_processing/error_report/error_report.png")

pred_csv.to_csv('vision_processing/result/result.csv',index=False)
print("성공 : {}, 실패 : {}, 성공률 : {}".format(round(cnt_total-error_cnt), error_cnt,round(100-error_cnt/cnt_total*100),3))