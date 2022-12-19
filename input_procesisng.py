import numpy as np
from cv2 import blur

def remove_background(MODEL_TYPE, color_img, depth_img):
    """return Array which background is removed. 

    Args:
        MODEL_TYPE (string): Method how to remove the background. blur, subsraction, greyabs included.
        color_img (ndarray): RGB image. Shape is W * H * 3. value covers 0~255 
        depth_img (ndarray): depth image. Shape is W * H * 1. value means depth distance as a milimeter

    Returns:
        ndarray: Image which backgorund removed. Shape is W * H * 3.
    """
    def compose(color_img, depth_img, filtered_img):
        composed_image = np.where(
            np.dstack([depth_img]*3) != 0, color_img, filtered_img)
        return composed_image

    depth_img = np.where(depth_img>=1000, 0, depth_img)

    if MODEL_TYPE == "blur9":
        filtered_img = blur(color_img, (9, 9))
        composed_img = compose(color_img, depth_img, filtered_img)
    elif MODEL_TYPE == "blur21":
        filtered_img = blur(color_img, (21, 21))
        composed_img = compose(color_img, depth_img, filtered_img)
    elif MODEL_TYPE == "blur100":
        filtered_img = blur(color_img, (100, 100))
        composed_img = compose(color_img, depth_img, filtered_img)
    # I think greyabs is not affective data augmentation way
    if MODEL_TYPE == "greyabs":
        composed_img = depth_img * 0.255
    elif MODEL_TYPE == "subtr":
        composed_img = compose(color_img, depth_img, 0)

    return composed_img
