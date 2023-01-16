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

from yolor.utils.google_utils import attempt_load
from yolor.utils.datasets import LoadStreams, LoadImages
from yolor.utils.general import (check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolor.utils.plots import plot_one_box
from yolor.utils.torch_utils import select_device, load_classifier, time_synchronized

from yolor.models.models import *
from yolor.utils.datasets import *
from yolor.utils.general import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(SOURCE, TARGET, BASE_NAME, model, device, opt):
    # SOURCE: background removal preprocessed input image, TARGET: original RGB image which will be showed with bounding box 
    half = device.type!= 'cpu'  # half precision only supported on CUDA
    pred_coord = []
    img0 = SOURCE
    auto_size = 64 # Use single image as input of "detect(SOURCE)"
    names = load_classes(opt.names)
    flower_color = (76,0,153)
    branch_color = (255,128,0)
    colors = [branch_color, flower_color]

    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None # run once 

    img = letterbox(img0, new_shape=opt.img_size, auto_size=auto_size)[0] # padded size

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16 or fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    s = ''
    with torch.no_grad():
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    for _, det in enumerate(pred):  
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
                    classNum2arr = np.array([int(cls)])
                    print('--')
                    print(classNum2arr)
                    coord = np.array(xywh)
                    pred_coord.append(np.concatenate((classNum2arr, coord)))

                    with open(txt_path + '.txt', 'a') as f:
                        # f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                        f.write(('%g ' * 6 + '\n') % (cls, conf, *xywh))  # label format
                    label = '%s %.2f' % (names[int(cls)], conf) # Add bbox to image
                    plot_one_box(xyxy, TARGET, label=label, color=colors[int(cls)], line_thickness=2)
    return pred_coord
    
