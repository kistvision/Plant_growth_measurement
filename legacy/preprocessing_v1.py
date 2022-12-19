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
    
print("number of dir 입력 :")
root_path = folders[int(input())]

rgb_dir = root_path + "/processed/original/rgb"
depth_dir = root_path + "/processed/original/depth"
color_depth_dir = root_path + "/processed/original/color_depth"

# flipped_rgb_dir = root_path + "/processed/flipped/rgb"
# flipped_depth_dir = root_path + "/processed/flipped/depth"
# flipped_color_depth_dir = root_path + "/processed/flipped/color_depth"

createFolder(rgb_dir)
createFolder(depth_dir)
createFolder(color_depth_dir)

# createFolder(flipped_rgb_dir)
# createFolder(flipped_depth_dir)
# createFolder(flipped_color_depth_dir)


Allfiles = os.listdir(root_path)
cnt = 0
for folder_file in Allfiles:
    search_dir = os.path.join(root_path, folder_file)
    name = folder_file.split("_")
    if len(name) > 1:
        date,_ = folder_file.split("_")
        # cnt += 1
        # n = str(cnt)
    if os.path.isdir(search_dir):
#         print("os ok")
        files = os.path.join(search_dir)
        files_dir = os.listdir(files)
        
#         print("files:", files)
        for file in files_dir:
            print("")
#             print('file ok')
#             print("file : ", file)
            path = os.path.join(files,file)
            # print(path)
            if 'RGB' in path:
                RGB_img = cv2.imread(path)
                # RGB_img_flipped = cv2.flip(RGB_img, 1)
                
                
                RGB_name = 'RGB_' + date + '_' + str(cnt) + '.png'
                # RGB_name_flipped = 'RGB_' + date + '_'  + str(cnt) + '_flipped'+ '.png'
                             

                save_dir = rgb_dir + '/' +RGB_name
                
                while os.path.isfile(save_dir):
                    cnt += 1
                    RGB_name = 'RGB_' + date + '_' + str(cnt) + '.png'
                    # RGB_name_flipped = 'RGB_' + date + '_'  + str(cnt) + '_flipped'+ '.png'
                    save_dir = rgb_dir + '/' +RGB_name
                print("현재 processing 이미지 :", RGB_name)
                    
                cv2.imwrite(save_dir, RGB_img)
                # cv2.imwrite(flipped_rgb_dir + '/' + RGB_name_flipped, RGB_img_flipped)
                
            elif 'Dist_img' in path:
                Dist_img = cv2.imread(path)
                # Dist_img_flipped = cv2.flip(Dist_img, 1)
                
                Dist_name = 'dist_' + date + '_' + str(cnt) + '.png'
                # Dist_name_flipped = 'dist_' + date + '_'  + str(cnt) + '_flipped_'+ '.png'
                
                save_dir = color_depth_dir + '/' +Dist_name
                
                while os.path.isfile(save_dir):
                    cnt += 1
                    Dist_name = 'dist_' + date + '_' + str(cnt) + '.png'
                    # Dist_name_flipped = 'dist_' + date + '_'  + str(cnt) + '_flipped_'+ '.png'
                    save_dir = color_depth_dir + '/' +Dist_name
                    
                cv2.imwrite(color_depth_dir + '/' + Dist_name, Dist_img)     
                # cv2.imwrite(flipped_color_depth_dir + '/' + Dist_name_flipped, Dist_img_flipped)           
                
            elif 'csv' in path:    
                Depth = pd.read_csv(path, header = None, low_memory = False)
                Depth = Depth[Depth<1].fillna(0)
                
                Depth_value = Depth.values
                # Depth_value_flipped = Depth_value.loc[:, ::-1]
                
                Depth_name = 'dist_' + date + '_' + str(cnt) + '.csv'
                # Depth_name_flipped = 'dist_' + date + '_'  + str(cnt) + '_flipped'+ '.csv'
                
                save_dir = depth_dir + '/' + Depth_name
                
                while os.path.isfile(save_dir):
                    cnt += 1
                    Depth_name = 'dist_' + date + '_' + str(cnt) + '.csv'
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

print("processing 완료")