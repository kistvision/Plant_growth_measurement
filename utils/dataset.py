import pandas as pd
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

def read_data(source):
    rgb_regix = f'{source}/rgb/*.png'
    depth_regix = f'{source}/depth/*.csv'
    rgb_files = sorted(glob(rgb_regix)) 
    depth_files = sorted(glob(depth_regix))

    rgb_images, depth_images, filenames = [], [], []
    cnt=0
    for(rgb_filename, depth_filename) in tqdm(zip(rgb_files, depth_files), total=len(rgb_files)):
        cnt += 1
        # if cnt > 20:
        #     break
        basename = os.path.basename(rgb_filename)
        color_img = cv2.imread(rgb_filename, cv2.IMREAD_COLOR)
        depth_df = pd.read_csv(depth_filename, header = None, low_memory = False)
        depth_val = depth_df.values
        depth_img = np.asanyarray(depth_val)
        rgb_images.append(color_img)
        depth_images.append(depth_img)
        filenames.append(basename)
    return rgb_images, depth_images, filenames
    
