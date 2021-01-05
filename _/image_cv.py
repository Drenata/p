import cv2 as cv
import numpy as np


def get_largest_contour(img):
    contours, _ = cv.findContours(
        (img * 255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    max_area = 0
    max_idx = -1

    for i in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[i])
        area = img[y : y + h, x : x + w].sum()
        if area > max_area:
            max_area = area
            max_idx = i

    return contours[max_idx]


def get_bounding_box(mask):
    x0, y0, w, h = cv.boundingRect(get_largest_contour(mask))

    x1 = x0 + w
    y1 = y0 + h

    return (x0, y0, x1, y1), (w, h)


def tight_crop(image):
    (x0, y0, x1, y1), (w, h) = get_bounding_box(image[:, :, 3])
    return image[y0:y1, x0:x1]
