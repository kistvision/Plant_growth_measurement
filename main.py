from utils.dataset import read_data
from yolor.models.models import Darknet
from yolor.utils.torch_utils import select_device, load_classifier, time_synchronized
from input_procesisng import *
from detect import detect
from estimate import get_plant_growth_info

import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./models/best_cucumber.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/testdata/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./outputs', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=448, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='./yolor/cfg/yolor_p6_custom.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='./data/farmbot.names', help='*.cfg path')
    parser.add_argument('--vertical_focal_len', type=float, default= 898.292, help='focal length of camera. default is L515')
    parser.add_argument('--horizontal_focal_len', type=float, default= 897.507, help='focal length of camera. default is L515')
    opt = parser.parse_args()
    print(opt)

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = Darknet(opt.cfg, opt.img_size).cuda()
    model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    rgb_imgs, depth_imgs, filenames = read_data(opt.source)

    success_cnt=0
    for rgb_img, depth_img, filename in zip(rgb_imgs, depth_imgs, filenames):
        composed = remove_background('blur100', rgb_img, depth_img)
        pred = detect(composed, rgb_img, filename, model, device, opt) 

        print(f'file : {filename}')
        if len(pred) < 2: # if number of predicted branche is one (less than two), we cannot calculate the distance between two branches
            continue
        success_cnt += 1
        get_plant_growth_info(rgb_img, depth_img, pred, filename, opt)
    print(f'success1:{success_cnt}; total: {len(rgb_imgs)}')
