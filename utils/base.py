import cv2
import os

def draw_text(img, 
              text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=1,
              font_thickness=1,
              text_color=(0, 25, 5, 0),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font,
                font_scale, text_color, font_thickness, cv2.LINE_AA)

    return text_size

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def folder_list(dirname):
    folders = []
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isdir(file_path):            
            folders.append(file_path)
    return sorted(folders)

def px2mm(px_value, depth_value, focalLen):
    mm_value = (px_value * depth_value)/focalLen
    return mm_value


def mm2px(mm_value, depth_value, focalLen):
    #depth value는 mm단위여야 함
    px_value = (mm_value * focalLen)/depth_value
    return px_value

def showme(img):
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sqrt(N):
    return N ** (1/2)

def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)