'''
yoloR에서 detect 한 만큼 이미지에서 detect 하기 위해 단독으로 사용되는 파일. 쓸 일이 없을것임 
'''
from yolor.models.models import Darknet
from yolor.utils.torch_utils import select_device, load_classifier, time_synchronized
from input_procesisng import *
from detect import detect
from utils.dataset import read_data
from estimate import crop_image

import argparse
import cv2
import torch

def binarize_depth(depth_img, threshold):
    depth_img = np.where(depth_img<threshold, depth_img, 0)
    depth_img = np.where(0<depth_img, 1, 0)
    depth_img = (depth_img*255).astype(np.uint8)

    kernel = np.ones((7,7), np.uint8)
    depth_img = cv2.morphologyEx(depth_img, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5,5), np.uint8)
    depth_img = cv2.morphologyEx(depth_img, cv2.MORPH_OPEN, kernel)

    return depth_img

def edge_depth(depth_img):
    canny = cv2.Canny(depth_img, 100, 200)
    return canny

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./models/best_cucumber.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./data/testdata', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='cropped', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=448, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='./yolor/cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='./data/farmbot_cucumber.names', help='*.cfg path')
    parser.add_argument('--vertical_focal_len', type=float, default= 898.292, help='focal length of camera. default is L515')
    parser.add_argument('--horizontal_focal_len', type=float, default= 1300.507, help='focal length of camera. default is L515')
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

    for rgb_img, depth_img, filename in zip(rgb_imgs, depth_imgs, filenames):
        composed = remove_background('blur100', rgb_img, depth_img)
        pred = detect(composed, filename, model, device, opt)

        if pred:
            print(filename)
            cropped_rgb = crop_image(rgb_img, pred[0], 0)
            cropped_depth = crop_image(depth_img.copy(), pred[0], 0)
            depth_threshhold = 0.9

            cropped_depth = binarize_depth(cropped_depth, depth_threshhold)
            canny = edge_depth(cropped_depth)
           
            print('cropped/RGB/'+'cropped_rgb'+filename[4:])
            cv2.imwrite('cropped/RGB/'+filename[4:], cropped_rgb)
            cv2.imwrite('cropped/depth/'+filename[4:], cropped_depth)
            cv2.imwrite('cropped/edge/'+filename[4:-4] + '_edge.png', canny)

