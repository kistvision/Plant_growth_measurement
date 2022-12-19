### 2022.04.25 (월)
########################################################
# Initialization
########################################################
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
########################################################
# Camera Capture
########################################################
def createRealsense_folder(fold_num):
    # Creating folder to store captured images from Intel Realsense
    # fold_num = 1
    
    curdate=datetime.today().strftime("%Y%m%d") # current year, month, and date
    dir_name=str("Img_%s_" %curdate + "%03d" %fold_num)

    try:
        while os.path.exists(dir_name):
            fold_num = fold_num + 1
            dir_name=str("Captured Image_" + 
                         "%s"%curdate + "/" + "Img_%s_" %curdate + "%03d" %fold_num)
        if not os.path.exists(dir_name):
            os.makedirs(os.path.join(dir_name))            
                
    except OSError:
        print("Failed to create directory !!")

    return dir_name
create10
########################################################
def cam_variables():

    ####### Camera settings #######
    # Setting about resoluation
    # w = 640
    # h = 480
    w = 1280
    h = 720

    # Define ROI(Region of Interest) to check whether current image is overexposed
    # ROI can change according to the loaction and size of target object 
    # Each point of rectangular-shaped ROI: (x1,y1), (x1,y2), (x2,y1), (x2,y2)
    x1 = int(w/7*3)
    x2 = int(w/7*4)
    y1 = int(h/3)
    y2 = int(h/3*2)

    x_range = [x1, x1]
    y_range = [y1, y2]

    center_area = int(w/7*h/3) #Region Of Interest for exposure time
    # The size of ROI

    # Define saturation area according to the saturation ratio
    sat_ratio = 5 #saturation area ratio (%)
    sat_area = int(center_area/100*sat_ratio) # saturation area pixel

    return w, h, x_range, y_range, sat_area

def initialization():

    w, h, _, _, _ = cam_variables()

    # Define variables 
    D435 = 0
    L515 = 1
    # initialize the number of capture
    step_num=0

    # CAMERA_TYPE = D435
    CAMERA_TYPE = L515 

    if CAMERA_TYPE == D435:
        w_depth = 1280
        h_depth = 720
    elif CAMERA_TYPE == L515:
        w_depth = 1024
        h_depth = 768

    # start time of running code
    t_ini = time.time()

    # initialize the status of Realsense
    pipeline = rs.pipeline()
    config = rs.config()
    #config.enable_device("832112073249")    ## Serial Number of RS1
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)  # Enable RGB image
    config.enable_stream(rs.stream.depth, w_depth, h_depth, rs.format.z16, 30)   # Enable depth image

    # start to receive the signal from Realsense
    profile = pipeline.start(config)
    dev = profile.get_device()

    # settings of color camera of Realsense
    dev.sensors[1].set_option(rs.option.enable_auto_exposure, 0)        # RGB camera Manual exposure: 0, Auto exposure: 1
    dev.sensors[1].set_option(rs.option.enable_auto_white_balance, 1)   # Auto White balance: 1
    dev.sensors[1].set_option(rs.option.brightness, 0)          # -64 ~ 64 , 0(default)
    dev.sensors[1].set_option(rs.option.contrast, 50)            #   0 ~ 64, 32

    if CAMERA_TYPE == D435:
        dev.sensors[1].set_option(rs.option.gamma, 500)             # 72 ~ 500, 100
    dev.sensors[1].set_option(rs.option.hue, 0)                # 0 ~ 128, 64
    dev.sensors[1].set_option(rs.option.saturation, 64)         # 0 ~ 128, 64
    dev.sensors[1].set_option(rs.option.sharpness, 50)         # 0 ~ 100, 50

    return pipeline, dev, t_ini

########################################################
# Capture
########################################################

def capture(pipeline, dev, t_ini,repeat_num):

    dir_name = createRealsense_folder(fold_num=1)

    avg_num = 3
    # iteration number of caputres to decrease the noise of images
    # after caputring, use this for acquire average image when target objects and Intel Realsense do not move
    # avg_num: 3~10, commonly avg_num = 3 is the BEST
    # [NOTICE] when either of the target or camera moves, use avg_num = 1

    # The number of images to be caputured in a single site
    # cap_num = 1
    cap_num = 3
    ####### Settings about wating times for Intel Realsensse #######
    time_delay1 = 1 # 1sec
    time_delay2 = 2 # 2sec

    w, h, x_range, y_range, sat_area = cam_variables()

    [x1, x2] = x_range
    [y1, y2] = y_range

    ####### Waiting time for ordinary  working of Realsense #######
 
    ###### in the outdoor and daytime environments
    # Set exps_time1 <= 78,  -> to decrease the time of capturing     
    # In the outdoor and daytime environments, even if using minimum time of exposure(78), overexposure may occur
    # In the case of overexposure, it is desirable to attach ND filter in front of the color camera.

    ###### in the indoor environments
    # Set exps_time1 >= 156, 
    # when the caputured images are too dark, exps_time should be increased

    exps_time = 312 #36, 78, 156, 312, 624, ...
    # our experiment site is outdoor environment: the smartfarm in Gangneung KIST

    # Restore the color and depth data from Realsense L515
    # after define the suitable exposure time as above-mentioned
    for step_num in range(0,cap_num):    
        step_num = step_num + 1
        
        dev.sensors[1].set_option(rs.option.exposure, exps_time)        # set initial exposure time for color camera

        time.sleep(time_delay2) # add time delay for white balance operation of color camera to work well
        
        color_arr = np.zeros((h,w,3),np.float64)

        # Capture the image as decreasing exposure time, while the color image does not exceed the sat_area
        sat_num = sat_area # 

        while sat_num >= sat_area :
            sat_num = 0
            
            print("expose time : %d" %(exps_time))
        
            # Acquire temporary color image
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            
            # Calculate the overexposed region from ROI, to check whether the overexposure occurs in the temporary image
            for c in range(x1,x2):
                for r in range(y1,y2):
                    # realsense1
                    if color_image[r,c,1] >= 250:
                        sat_num = sat_num + 1

            # If the overexposed region in ROI is larger than the threshold(sat_area), let the exposure time downsize as half
            # else: use the same exposure time value
            if sat_num >= sat_area:
                exps_time = exps_time/2
                
            
            # 새롭게 정의된 노출시간에 맞게 리얼센스 재설정
            # Set the new variables of camera again, reflecting the newly define exposure time
            dev.sensors[1].set_option(rs.option.exposure, exps_time) # realsense1
            
            # Add time delay to make auto function of camera such as white balance
            time.sleep(time_delay1)

        #########################
        # Here is the start of steps to acquire color image
        # the camera caputures the images according to "avg_num", and calculates the average image 
        #########################
        avg_ind = 0
        for avg_ind in range(0,avg_num):
            avg_ind=avg_ind+1
            
            # realsense1 rgb image with time averaging
            frames1 = pipeline.wait_for_frames()        
            color_frame = frames1.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            color_arr = color_arr + np.array(color_image,dtype=np.float64)/avg_num
            
        ## restore color image
        color_arr = np.array(np.round(color_arr),dtype=np.uint8)
        BASE_NAME = "%s/RGB_%d.png" %(dir_name, int(step_num)+repeat_num)
        cv2.imwrite(BASE_NAME,color_arr)

        ## capture depth image to acquire depth information
        frames = pipeline.wait_for_frames() # 

        align_to = rs.stream.color
        align = rs.align(align_to)
        processed = align.process(frames)
        depth_frame = processed.get_depth_frame()
        
        dist = np.zeros((h,w)) # initialize array to restore distance information
        
        for y in range(h):
            for x in range(w):
                zDepth = depth_frame.get_distance(int(x),int(y))
                dist[y, x] = zDepth # distance value from each pixel (x, y)
                #print(format(zDepth,".1f"), end =" ")
            #print('\n')

        colorizer = rs.colorizer() # restore depth image as png and depth data as csv
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        BASE_NAME2 = "%s/Dist_img_%d.png" %(dir_name, int(step_num)+repeat_num)
        cv2.imwrite(BASE_NAME2, colorized_depth)
    
        depth_image = np.asanyarray(dist)
        BASE_NAME3 = "%s/Dist_data_%d.csv" %(dir_name, int(step_num)+repeat_num)
        np.savetxt(BASE_NAME3, depth_image, delimiter=',', fmt='%f')
        depth_value = depth_image.astype("uint8")

        t_cap = time.time()
        elapsed = round(t_cap - t_ini , 2) # total time for working
        # print("Finished #%d th images, %.2f sec" %(int(step_num), elapsed))  
        
        time.sleep(time_delay2)

        # return csv_name
        # return depth_name
        return color_arr, depth_value, colorized_depth, BASE_NAME, BASE_NAME2, BASE_NAME3

def finalize(pipeline):
    ### de-connect
    pipeline.stop()
    print ("Process End")
    


#####################################################
#####################################################
# Main for Online Processing
#####################################################
#####################################################

while True:
    GO = int(input("진행(1) / 종료(2) :"))
    
    if GO == 1:
        # del (PIPELINE)
        PIPELINE, DEV, T_INI = initialization()

        # 1st capture
        RGB1, DEPTH1,COLOR_DEPTH1, BASE_NAME, BASE_NAME2, BASE_NAME3 = capture(PIPELINE, DEV, T_INI,0)
        # getProcessing(RGB1, DEPTH1,COLOR_DEPTH1)

        # # 2nd capture
        # RGB2, DEPTH2, COLOR_DEPTH2, BASE_NAME, BASE_NAME2, BASE_NAME3 = capture(PIPELINE, DEV, T_INI,1)
        # getProcessing(RGB2, DEPTH2, COLOR_DEPTH2)

        # # 3rd capture
        # RGB3, DEPTH3, COLOR_DEPTH3, BASE_NAME, BASE_NAME2, BASE_NAME3 = capture(PIPELINE, DEV, T_INI,2)
        # getProcessing(RGB3, DEPTH3, COLOR_DEPTH3)
        
        finalize(PIPELINE)
    elif GO == 2:
        break
    else:
        print("error")