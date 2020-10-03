#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image


class CvDrawText:
    def __init__(self):
        pass

    @classmethod
    def puttext(cls,
                cv_image,
                text,
                point,
                font_path,
                font_size,
                color=(0, 0, 0)):
        font = ImageFont.truetype(font_path, font_size)

        cv_rgb_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_rgb_image)

        draw = ImageDraw.Draw(pil_image)
        draw.text(point, text, fill=color, font=font)

        cv_rgb_result_image = np.asarray(pil_image)
        cv_bgr_result_image = cv.cvtColor(cv_rgb_result_image,
                                          cv.COLOR_RGB2BGR)

        return cv_bgr_result_image
