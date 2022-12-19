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

folders_dir = sorted(os.listdir(root_path))
#         print("files:", files)

cnt_rgb = 0
cnt_cdepth = 0
cnt_depth = 0
date = "20220621"
for folder in folders_dir:
#             print('file ok')
#             print("file : ", file)
    path = os.path.join(root_path,folder)
    if 'raw' in path:
        pass
    else: 
        print("\n현재 입력 :", path)
        files = os.listdir(path)
        # print(files_dir)
        for file in files:
            file_dir = os.path.join(path,file)
            if 'RGB' in file_dir:
                # print("\n입력에 해당하는 process : RGB")
                RGB_img = cv2.imread(file_dir)        
                RGB_name = 'RGB_' + date + '_' + str(cnt_rgb) + '.png'
                save_dir = rgb_dir + '/' +RGB_name
                # print(save_dir)
                while os.path.isfile(save_dir):
                    cnt_rgb += 1
                    RGB_name = 'RGB_' + date + '_' + str(cnt_rgb) + '.png'
                    save_dir = rgb_dir + '/' +RGB_name
                # print("현재 processing 이미지 :", RGB_name)
                cv2.imwrite(save_dir, RGB_img)     
                
                print("\n#####################################")
                print("\n RGB 출력 : ", str(save_dir))
                    
            elif ('Dist_img' in file_dir or 'depth_aligned' in file_dir) and 'png' in file_dir:
                # print("\n입력에 해당하는 process : color depth")
                
                Dist_img = cv2.imread(file_dir)
                
                Dist_name = 'dist_' + date + '_' + str(cnt_cdepth) + '.png'
                
                save_dir = color_depth_dir + '/' +Dist_name
                
                while os.path.isfile(save_dir):
                    cnt_cdepth += 1
                    Dist_name = 'dist_' + date + '_' + str(cnt_cdepth) + '.png'
                    save_dir = color_depth_dir + '/' +Dist_name
                    
                cv2.imwrite(color_depth_dir + '/' + Dist_name, Dist_img) 
                
                print("\n Depth Img 출력 : ", str(save_dir))
                
            elif 'csv' in file_dir:   
                # print("\n입력에 해당하는 process : depth")
                Depth = pd.read_csv(file_dir, header = None, low_memory = False)
                Depth_value = Depth.values
                
                Depth_name = 'dist_' + date + '_' + str(cnt_depth) + '.csv'
                
                save_dir = depth_dir + '/' + Depth_name
                
                while os.path.isfile(save_dir):
                    cnt_depth += 1
                    Depth_name = 'dist_' + date + '_' + str(cnt_depth) + '.csv'
                    save_dir = depth_dir + '/' + Depth_name
                    
                Depth.to_csv(depth_dir + '/' + Depth_name, header = None, index = False)
                print("\n Depth CSV 출력 : ", str(save_dir))
                print("\n#####################################")

print("\n#####################################")
print("\nResult check")

print("\nRGB :", RGB_img.shape)

print("\nColor Depth img :", Dist_img.shape)

print("\nDepth csv :",Depth.shape)

print("\nRGB : %d개, Color Depth : %d개, csv Depth : %d"%(cnt_rgb, cnt_cdepth, cnt_depth))
print("\n#####################################")


print(cnt_rgb, "개의 이미지 processing 완료")