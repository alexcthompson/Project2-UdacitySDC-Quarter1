import numpy as np
import cv2
import IPython.display
import PIL.Image
from io import BytesIO


# for image modification
import random
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform


def showarray(a, fmt='png', width=None, height=None):
    '''
    Displays an image without the ugliness of matplotlib
    '''
    a = np.uint8(a)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue(), width=width, height=height))


def equalize_rgb(img, clahe=True):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(3, 3))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

    else:
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return img_output


def equalize_set(set, clahe=True):
    return np.array([equalize_rgb(image, clahe=clahe) for image in set])


def pixel_normalization(p):
    return (p - 128.) / 128


# This code borrowed from https://navoshta.com/traffic-signs-classification/,
# a previous term student who worked out the gritty details of transforming images
def rotate_ims(X, intensity):
    for i in range(X.shape[0]):
        delta = 30. * intensity  # scale using augmentation intensity
        X[i] = rotate(X[i], random.uniform(-delta, delta), mode = 'edge')
    return X


def apply_projection_transform(X, intensity):
    image_size = X.shape[1]
    d = image_size * 0.3 * intensity
    for i in range(X.shape[0]):
        tl_top = random.uniform(-d, d)     # Top left corner, top margin
        tl_left = random.uniform(-d, d)    # Top left corner, left margin
        bl_bottom = random.uniform(-d, d)  # Bottom left corner, bottom margin
        bl_left = random.uniform(-d, d)    # Bottom left corner, left margin
        tr_top = random.uniform(-d, d)     # Top right corner, top margin
        tr_right = random.uniform(-d, d)   # Top right corner, right margin
        br_bottom = random.uniform(-d, d)  # Bottom right corner, bottom margin
        br_right = random.uniform(-d, d)   # Bottom right corner, right margin

        transform = ProjectiveTransform()
        transform.estimate(
            np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )),
            np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
        X[i] = warp(X[i], transform, output_shape=(image_size, image_size),
                    order = 1, mode = 'edge')

    return X
