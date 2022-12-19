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
    if "2022" in name:
        print("{} : {}".format(name, i))
    
print("number of dir 입력 :")
root_path = folders[int(input())]

depth_dir = root_path + "/depth"

# flipped_rgb_dir = root_path + "/processed/flipped/rgb"
flipped_depth_dir = root_path + "/processed/flipped/depth"
# flipped_color_depth_dir = root_path + "/processed/flipped/color_depth"

# createFolder(depth_dir)

# createFolder(flipped_rgb_dir)
createFolder(flipped_depth_dir)
# createFolder(flipped_color_depth_dir)

cnt = 0

search_dir = os.path.join(root_path, depth_dir)

files_dir = os.listdir(search_dir)

#         print("files:", files)
for file in files_dir:
    Depth = pd.read_csv(file, header = None, low_memory = False)
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

print("\nDepth csv :",Depth.shape)

print("\n#####################################")

print("processing 완료")