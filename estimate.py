from utils.base import *
import numpy as np

def get_plant_growth_info(rgb_image, depth_image, bounding_boxes, filename, opt):
    '''
    Measure stem diameter and the distance between a branch point and another branch point
    based on image and coordinates of detected branch points.
    '''
    COLOR_B2BLINE = (0,255,204)
    weighted_bounding_boxes = [weight_coord(bb, 1280, 720) for bb in bounding_boxes]
    sorted_bounding_boxes = sort_boxes(weighted_bounding_boxes)
    annotate_box(weighted_bounding_boxes, rgb_image)
    lowest_branch = sorted_bounding_boxes[0]
    second_low_branch = sorted_bounding_boxes[1]


    dist = branch_distance(lowest_branch, second_low_branch, rgb_image, depth_image, opt)
    diameter = stem_diameter(lowest_branch, rgb_image, depth_image, opt)

    distance_text = "Upper Branch Point to Lower Branch Point : " + str(round(dist,3))  + "mm"
    diameter_text = "Stem Diameter: "+str(round(diameter,3))+"mm"
    draw_text(rgb_image, distance_text, text_color=COLOR_B2BLINE, pos=(100,50), font=cv2.FONT_HERSHEY_SIMPLEX)
    draw_text(rgb_image, diameter_text, text_color=COLOR_B2BLINE, pos=(100,150), font=cv2.FONT_HERSHEY_SIMPLEX)

    # Check alignment of rgb and depth image with overlapping them
    # depth_image_ = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    # depth_image = depth_image_.astype('uint8')
    # de = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
    # fusion = cv2.addWeighted(rgb_image, 0.4, de, 0.6, 0)#dtype=cv2.CV_32F)
    # cv2.imshow("rgb+depth overlap", fusion)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # cv2.imshow("di", depth_image)
    cv2.imshow("ci", rgb_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    
def branch_distance(box_branch1, box_branch2, rgb_image, depth_image, opt):
    '''
    Calculate distance bewteen two bounding boxes of branch point class
    '''
    _, x1, y1, w1, h1 = box_branch1
    _, x2, y2, w2, h2 = box_branch2

    cropped_depth1 = crop_image(depth_image, box_branch1, 5)
    cropped_depth2 = crop_image(depth_image, box_branch2, 5)

    d1 = np.nanmedian(np.where(cropped_depth1==0.0, np.nan, cropped_depth1)) * 1000
    d2 = np.nanmedian(np.where(cropped_depth2==0.0, np.nan, cropped_depth2)) * 1000

    bb_px_dist = sqrt((x1-x2)**2 + (y1-y2)**2)
    bb_mm_dist = px2mm(bb_px_dist, d1, opt.vertical_focal_len)
    dist = bb_mm_dist
    cv2.line(rgb_image, (x1, y1),(x2, y2),(0,255,255), 2)
    return dist

def stem_diameter(box_branch, rgb_image, depth_image, opt):
    LOW_RATE = 0.2
    HIGH_RATE = 0.5
    STEMLINE_COLOR = (208,205,195)
    STEM_COLOR = (0,255,204)

    _, x, y, w, h = box_branch # x, y: center point of bounding box
    rectx, recty = x-w//2, y-h//2 # coordinate of top-left
    MIN_ROI_H = h*LOW_RATE
    MAX_ROI_H = h*HIGH_RATE

    crop_rgb, crop_depth = crop_image(rgb_image, box_branch), crop_image(depth_image, box_branch)
    to_estimate_image = estimation_image(crop_rgb, crop_depth, 'rgbmean') # options for measuring stem diameter: 'rgbmean', 'depth'
    roi_median_depth = np.nanmedian(np.where(crop_depth==0.0, np.nan, crop_depth))
    estimate_pointY = h//2 - round((MIN_ROI_H+MAX_ROI_H)/2) # estimate at the center point from ROI of branch point box

    stem_start, stem_end = max_cluster(to_estimate_image[estimate_pointY][0:w]) # find the indices of the longest stem cluster
    stem_start += rectx
    stem_end += rectx

    diameter = px2mm(stem_end-stem_start, roi_median_depth* 1000, opt.horizontal_focal_len) 
    # convert number of pixel to mm-unit length

    cv2.line(rgb_image, (rectx, recty+estimate_pointY), (rectx+w, recty+estimate_pointY), STEMLINE_COLOR, 2)
    cv2.line(rgb_image, (stem_start, recty+estimate_pointY), (stem_end, recty+estimate_pointY), STEM_COLOR, 2)

    return diameter

def estimation_image(crop_rgb, crop_depth, method):
    # compensating depth values
    if method == 'depth':
        ret = recover_depth(crop_depth)
    elif method == 'rgbmean':
        rd = recover_depth(crop_depth)
        ret = recover_rgb(crop_rgb, rd)
    return ret

def recover_depth(crop_depth):
    THRESHOLD = 100
    crop_depth = crop_depth.copy()
    median_depth = np.nanmedian(np.where(crop_depth==0.0, np.nan, crop_depth)) *  1000 # get 1000 times values to convert meter to milimeter
    for i in range(crop_depth.shape[0]):
        for j in range(crop_depth.shape[1]):
            pixel_value = crop_depth[i][j] * 1000
            # roi_depth_uint16 = median_depth.astype("uint8")
            if pixel_value>(median_depth-THRESHOLD) and pixel_value<(median_depth+THRESHOLD):
                crop_depth[i,j] = 255
            else:
                crop_depth[i,j] = 0
    return crop_depth

def recover_rgb(crop_rgb, recovered_crop_depth):
    # get average pixel intensity of R, G, B from cropped image of branch point 
    # recover depth values based on the RGB intensity
    SCALE = 1
    save_color = np.zeros(shape=(crop_rgb.shape[0],crop_rgb.shape[1],crop_rgb.shape[2]), dtype = np.uint8)

    for k in range(crop_rgb.shape[2]):
        for i in range(crop_rgb.shape[0]):
            for j in range(crop_rgb.shape[1]):
                if recovered_crop_depth[i,j] == 255:
                    save_color[i,j,k] = crop_rgb[i,j,k]
                else:
                    save_color[i,j,k] = 0
    
    median_blue = np.nanmedian(crop_rgb[:,:,0][save_color[:,:,0]>0])
    median_green= np.nanmedian(crop_rgb[:,:,1][save_color[:,:,1]>0])
    median_red = np.nanmedian(crop_rgb[:,:,2][save_color[:,:,2]>0])

    thresholdB = np.std(crop_rgb[:,:,0])*SCALE
    thresholdG = np.std(crop_rgb[:,:,1])*SCALE
    thresholdR = np.std(crop_rgb[:,:,2])*SCALE
    threshold_color_img = np.zeros(shape=(crop_rgb.shape[0],crop_rgb.shape[1],crop_rgb.shape[2]), dtype = np.uint8)

    for i in range(crop_rgb.shape[0]):
        for j in range(crop_rgb.shape[1]):
            pixel_b = crop_rgb[i,j,0]
            pixel_g = crop_rgb[i,j,1]
            pixel_r = crop_rgb[i,j,2]
            if pixel_b > (median_blue-thresholdB) and pixel_b < (median_blue+thresholdB):
                if pixel_g > (median_green-thresholdG) and pixel_g < (median_green+thresholdG):
                    if pixel_r > (median_red-thresholdR) and pixel_r < (median_red+thresholdR):
                        threshold_color_img[i,j,0] = pixel_b
                        threshold_color_img[i,j,1] = pixel_g
                        threshold_color_img[i,j,2] = pixel_r
                    else:
                        threshold_color_img[i,j,0] = 0
                        threshold_color_img[i,j,1] = 0
                        threshold_color_img[i,j,2] = 0

    threshold_gray_img= cv2.cvtColor(threshold_color_img, cv2.COLOR_BGR2GRAY)
    ret,c2b_img_cv2 = cv2.threshold(threshold_gray_img,10,255,cv2.THRESH_BINARY)
    closed_img = close(c2b_img_cv2)

    return closed_img

def close(image):
    neg_image = 255 - image
    kernel = np.ones((3,3), np.uint8)
    dil = cv2.dilate(neg_image, kernel, iterations=1)
    ero = cv2.erode(dil, kernel, iterations=1)
    ret = 255 - ero
    return ret

def max_cluster(estimate_line, threshold=0.1):
    end_point, max_adjacent_pixels = 0,0
    cur_adjacent_pixels = 0
    for i in range(1, len(estimate_line)):
        if abs(estimate_line[i]-estimate_line[i-1]) >= threshold:
            if cur_adjacent_pixels > max_adjacent_pixels:
                max_adjacent_pixels = cur_adjacent_pixels
                end_point = i
            cur_adjacent_pixels = 0
        else:
            cur_adjacent_pixels += 1
    if cur_adjacent_pixels > max_adjacent_pixels:
        end_point = i
        max_adjacent_pixels = cur_adjacent_pixels
    start_point = end_point-max_adjacent_pixels-1
    return start_point, end_point

def annotate_box(bounding_boxes, rgb_image):
    CENTOR_COLOR = (255,0,255)
    for box in bounding_boxes:
        _, x, y, w, h = box
        cv2.circle(rgb_image, (x, y), 2, CENTOR_COLOR, 2)

def sort_boxes(bounding_boxes):
    return sorted(bounding_boxes, key=lambda x: x[2], reverse=True) # sort by y, desc

def weight_coord(coord, full_w, full_h):
    # convert YOLO format coordinates to absolute pixel coordinates in the image
    c, x, y, w, h = coord
    Rx,Rw = tuple(round(x*full_w) for x in (x,w))
    Ry,Rh = tuple(round(x*full_h) for x in (y,h))
    return c, Rx, Ry, Rw, Rh

def crop_image(image, box, crop_size=0):
    _, x, y, w, h = box
    if crop_size == 0:
        return image[y - h//2 : y + h//2, x - w//2 : x + w//2]
    else:
        return image[y-crop_size : y+crop_size, x-crop_size: x+crop_size] 
