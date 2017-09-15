#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
#Created by r05922145@ntu.edu.tw at 2017/09/15

import sys
from PIL import Image

def half_img_rgb(path):
    img =  Image.open(path)
    size = img.size
    new_img =  Image.new('RGB', size)
    pixel = img.load()
    new_pixel = new_img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            r, g, b = pixel[x, y]
            new_pixel[x, y] = int(r/2), int(g/2), int(b/2)
    new_img.save('Q2.png')

if __name__ == '__main__' :
    path =  sys.argv[1]
    half_img_rgb(path)
