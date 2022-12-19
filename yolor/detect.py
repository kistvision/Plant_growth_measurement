import sys 
import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from .utils.google_utils import attempt_load
from .utils.datasets import LoadStreams, LoadImages
from .utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from .utils.plots import plot_one_box
from .utils.torch_utils import select_device, load_classifier, time_synchronized

from .models.models import *
from .utils.datasets import *
from .utils.general import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(SOURCE, BASE_NAME, model, device):
   # temporary configs
    DEVICE = ""
    WEIGHTS = "./models/best_cucumber.pt"
    IMGSZ = 448
    AUGMENT = False
    CONF_THRES = 0.2
    IOU_THRES = 0.5
    CLASSES = None
    AGNOSTIC_NMS = False
    BASE_NAME = ""
    NAMES="./data/farmbot_cucumber.names"
    #YOLOCFG: "./yolor/cfg/yolor_p6.cfg" -> tomato
    CFG = "./yolor/cfg/yolor_p6.cfg"


    half = device.type != 'cpu'  # half precision only supported on CUDA
    pred_coord = []
    img0 = SOURCE


    # Use single image as input of "detect(SOURCE)"
    img_size = IMGSZ
    auto_size = 64

    # Get names and colors
    names = load_classes(NAMES)
    # 변경2 
    flower_color = (76,0,153)    # Img 상에 결과 출력할때 위한 글씨 색, 이후 plot_one_box 에서 사용
    branch_color = (255,128,0)   # Img 상에 결과 출력할때 위한 글씨 색
    # for tomato
    # 변경 3
    # if N == 1:
    colors = [flower_color, branch_color]
    # elif N == 2:
    # for cucumber
        # colors = [branch_color]
    img = torch.zeros((1, 3, IMGSZ, IMGSZ), device=device)  # init img
    # print("===============")
    # print(type(img))
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    # print("===============")
    # Padded resize
    img = letterbox(img0, new_shape=img_size, auto_size=auto_size)[0]

    # cv2.imshow("im", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16 or fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # t1 = time_synchronized()
    s = ''
    with torch.no_grad():
        pred = model(img, augment=AUGMENT)[0]

    # Apply NMS
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
    # print("pred: ", pred)
    # t2 = time_synchronized()


    # Process detections
    for i, det in enumerate(pred):  
        # normalization gain whwh
        save_path = "inference/output/" + BASE_NAME
        txt_path = "inference/output/" + Path(BASE_NAME).stem
        s += '%gx%g ' % img.shape[2:]  # print string

        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            # print(det[:,-1])
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

                    with open(txt_path + '.txt', 'a') as f:
                        # f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                        f.write(('%g ' * 6 + '\n') % (cls, conf, *xywh))  # label format
                    label = '%s %.2f' % (names[int(cls)], conf) # Add bbox to image
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)
                    # print(save_path)
                    # cv2.imshow("im", img0)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                    # cv2.imwrite(save_path+'1.png', img0)

    return pred_coord
    