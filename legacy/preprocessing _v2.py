from email import header
from PIL import Image #image 파일 열기
import os #주소로 폴더 열기
import cv2 as cv2
import glob
import shutil
import csv
import pandas as pd
from pandas import read_csv

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
        
def folder_list(dirname):
    folders = []
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isdir(file_path):            
            folders.append(file_path)
    return sorted(folders)

folders = folder_list('./')

for i,name in enumerate(folders):
    if "Data2022" in name:
        print("{} : {}".format(name, i))
    
# print()

root_path = folders[int(input("number of dir 입력 :"))]

rgb_dir = root_path + "/processed/original/rgb"
depth_dir = root_path + "/processed/original/depth"
color_depth_dir = root_path + "/processed/original/color_depth"

createFolder(rgb_dir)
createFolder(depth_dir)
createFolder(color_depth_dir)


# flipped_rgb_dir = root_path + "/processed/flipped/rgb"
# flipped_depth_dir = root_path + "/processed/flipped/depth"
# flipped_color_depth_dir = root_path + "/processed/flipped/color_depth"

files_dir = os.listdir(root_path)
#         print("files:", files)
cnt_rgb = 0
cnt_cdepth = 0
cnt_depth = 0
date = str(20220520)
for file in files_dir:
#             print('file ok')
#             print("file : ", file)
    path = os.path.join(root_path,file)
    # cnt += 1
    # print(path)
    if 'RGB' in path:
        RGB_img = cv2.imread(path)
        RGB_img_flipped = cv2.flip(RGB_img, 1)
        
        
        RGB_name = 'RGB_' + date + '_' + str(cnt_rgb) + '.png'
        # RGB_name_flipped = 'RGB_' + date + '_'  + str(cnt) + '_flipped'+ '.png'
                        

        save_dir = rgb_dir + '/' +RGB_name
        
        while os.path.isfile(save_dir):
            cnt_rgb += 1
            RGB_name = 'RGB_' + date + '_' + str(cnt_rgb) + '.png'
            # RGB_name_flipped = 'RGB_' + date + '_'  + str(cnt) + '_flipped'+ '.png'
            save_dir = rgb_dir + '/' +RGB_name
            
        print("현재 processing 이미지 :", RGB_name)
            
        cv2.imwrite(save_dir, RGB_img)
        # cv2.imwrite(flipped_rgb_dir + '/' + RGB_name_flipped, RGB_img_flipped)
        
    elif ('Dist_img' in path or 'depth_aligned' in path) and 'png' in path:
        if 'raw' in path:
            pass
        else:
            Dist_img = cv2.imread(path)
            # Dist_img_flipped = cv2.flip(Dist_img, 1)
            
            Dist_name = 'dist_' + date + '_' + str(cnt_cdepth) + '.png'
            # Dist_name_flipped = 'dist_' + date + '_'  + str(cnt) + '_flipped_'+ '.png'
            
            save_dir = color_depth_dir + '/' +Dist_name
            
            while os.path.isfile(save_dir):
                cnt_cdepth += 1
                Dist_name = 'dist_' + date + '_' + str(cnt_cdepth) + '.png'
                # Dist_name_flipped = 'dist_' + date + '_'  + str(cnt) + '_flipped_'+ '.png'
                save_dir = color_depth_dir + '/' +Dist_name
                
            cv2.imwrite(color_depth_dir + '/' + Dist_name, Dist_img)     
            # cv2.imwrite(flipped_color_depth_dir + '/' + Dist_name_flipped, Dist_img_flipped)           
        
    elif 'csv' in path:   
        if 'raw' in path:
            pass
        else: 
            Depth = pd.read_csv(path, header = None, low_memory = False)
            Depth_value = Depth.values
            # Depth_value_flipped = Depth_value.loc[:, ::-1]
            
            Depth_name = 'dist_' + date + '_' + str(cnt_depth) + '.csv'
            # Depth_name_flipped = 'dist_' + date + '_'  + str(cnt) + '_flipped'+ '.csv'
            
            save_dir = depth_dir + '/' + Depth_name
            
            while os.path.isfile(save_dir):
                cnt_depth += 1
                Depth_name = 'dist_' + date + '_' + str(cnt_depth) + '.csv'
                # Depth_name_flipped = 'dist_' + date + '_'  + str(cnt) + '_flipped'+ '.csv'
                save_dir = depth_dir + '/' + Depth_name
                
            Depth.to_csv(depth_dir + '/' + Depth_name, header = None, index = False)
            # Depth_value.to_csv(flipped_depth_dir + '/' + Depth_name_flipped, index = False)

print("\n#####################################")
print("\nresolution check")

print("\nRGB :", RGB_img.shape)

print("\nColor Depth img :", Dist_img.shape)

print("\nDepth csv :",Depth.shape)

print("\n#####################################")


print(cnt_rgb, "개의 이미지 processing 완료")