# @Author: Pavlo Butenko

from enum import Enum
from typing import Type

import cv2
import numpy as np


def enum_from_string(value: str, enum: Type[Enum]):
    """
    Convert a string to an enum value, case insensitive.
    
    Arguments:
        value: The string value to convert.
        enum: The type of enum class to convert to.
    """
    enum_value = [evalue for evalue in enum if evalue.name.lower() == value.lower()]
    if len(enum_value) == 0:
        raise Exception("Not supported")
    return enum_value[0]

def get_resize_df(img: cv2.Mat, df: int) -> tuple[int, int]:
    """
    Resize the image to the nearest multiple of df.

    Arguments:
        img: The image to resize.
        df: The divisor factor.

    Returns:
        A tuple containing the new height and width of the image.
    """
    h, w = img.shape[:2]
    new_sizes = list(map(lambda x: int(x // df * df), [h, w]))
    return new_sizes[0], new_sizes[1]

def prepare_image(img: cv2.Mat, df: int) -> tuple[cv2.Mat, list[int]]:
    """
    Prepare the image for processing by resizing it to the nearest multiple of df.

    Arguments:
        img: The image to prepare.
        df: The divisor factor.

    Returns:
        A tuple containing the resized image and a list of scaling factors for width and height.
    """
    h, w = img.shape[:2]
    new_h, new_w = get_resize_df(img, df)
    if h != new_h or w != new_w:
        img = cv2.resize(np.copy(img), (new_w, new_h))
    return img, np.array([w / new_w, h / new_h])