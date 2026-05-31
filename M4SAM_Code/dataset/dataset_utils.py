import os, sys
from PIL import Image

import random
import numpy as np
from PIL import ImageEnhance
import torch

#several data augumentation strategies
def cv_random_flip_img(imgs, labels, depths):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i, img in enumerate(imgs):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
            depths[i] = depths[i].transpose(Image.FLIP_LEFT_RIGHT)
            if labels[i] is not None:
                labels[i] = labels[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, labels, depths

def randomCrop_img(imgs, labels, depths):
    border=30
    image_width = imgs[-1].size[0]
    image_height = imgs[-1].size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    for i, img in enumerate(imgs):
        imgs[i] = imgs[i].crop(random_region)
        depths[i] = depths[i].crop(random_region)
        if labels[i] is not None:
            labels[i] = labels[i].crop(random_region)
    return imgs, labels, depths

def randomRotation_img(imgs, labels, depths):
    rand_count = random.random()
    random_angle = np.random.randint(-15, 15)
    mode=Image.BICUBIC
    if rand_count>0.8:
        for i, img in enumerate(imgs):
            imgs[i] = imgs[i].rotate(random_angle, mode)
            depths[i] = depths[i].rotate(random_angle, mode)
            if labels[i] is not None:
                labels[i] = labels[i].rotate(random_angle, mode)
    return imgs, labels, depths

def colorEnhance_img(imgs):
    bright_intensity=random.randint(5, 15) / 10.0
    contrast_intensity = random.randint(5, 15) / 10.0
    color_intensity = random.randint(0, 20) / 10.0
    sharp_intensity = random.randint(0, 30) / 10.0
    for i, img in enumerate(imgs):
        imgs[i]=ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        imgs[i]=ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        imgs[i]=ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        imgs[i]=ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    return imgs

def clipper(abs_path,video_length):

    return [abs_path[i:i+video_length] for i in range(0,len(abs_path),video_length)]

    

'''FOR Video 自行按照参数重载，不作私有函数'''
def cv_random_flip(imgs, labels, depths,video_rgb,video_gt,video_dep):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i, img in enumerate(imgs):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
            depths[i] = depths[i].transpose(Image.FLIP_LEFT_RIGHT)
            if labels[i] is not None:
                labels[i] = labels[i].transpose(Image.FLIP_LEFT_RIGHT)
        for i,img in enumerate(video_rgb):
            video_rgb[i]=img.transpose(Image.FLIP_LEFT_RIGHT)
            video_gt[i]=video_gt[i].transpose(Image.FLIP_LEFT_RIGHT)
            video_dep[i]=video_dep[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, labels, depths,video_rgb,video_gt,video_dep

def randomCrop(imgs, labels, depths,video_rgb,video_gt,video_dep):
    border=30
    image_width = imgs[-1].size[0]
    image_height = imgs[-1].size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    for i, img in enumerate(imgs):
        imgs[i] = imgs[i].crop(random_region)
        depths[i] = depths[i].crop(random_region)
        if labels[i] is not None:
            labels[i] = labels[i].crop(random_region)
    for i,img in enumerate(video_rgb):
        video_rgb[i] = video_rgb[i].crop(random_region)
        video_gt[i] = video_gt[i].crop(random_region)
        video_dep[i] = video_dep[i].crop(random_region)

    return imgs, labels, depths,video_rgb,video_gt,video_dep

def randomRotation(imgs, labels, depths,video_rgb,video_gt,video_dep):
    rand_count = random.random()
    random_angle = np.random.randint(-15, 15)
    mode=Image.BICUBIC
    if rand_count>0.8:
        for i, img in enumerate(imgs):
            imgs[i] = imgs[i].rotate(random_angle, mode)
            depths[i] = depths[i].rotate(random_angle, mode)
            if labels[i] is not None:
                labels[i] = labels[i].rotate(random_angle, mode)

        for i,img in enumerate(video_rgb):
            video_rgb[i] = video_rgb[i].rotate(random_angle, mode)
            video_gt[i] = video_gt[i].rotate(random_angle, mode)
            video_dep[i] = video_dep[i].rotate(random_angle, mode)

    return imgs, labels, depths,video_rgb,video_gt,video_dep

def colorEnhance(imgs,video_rgb):
    bright_intensity=random.randint(5, 15) / 10.0
    contrast_intensity = random.randint(5, 15) / 10.0
    color_intensity = random.randint(0, 20) / 10.0
    sharp_intensity = random.randint(0, 30) / 10.0
    for i, img in enumerate(imgs):
        imgs[i]=ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        imgs[i]=ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        imgs[i]=ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        imgs[i]=ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    for i,img in enumerate(video_rgb):
        video_rgb[i]=ImageEnhance.Brightness(video_rgb[i]).enhance(bright_intensity)
        video_rgb[i]=ImageEnhance.Contrast(video_rgb[i]).enhance(contrast_intensity)
        video_rgb[i]=ImageEnhance.Color(video_rgb[i]).enhance(color_intensity)
        video_rgb[i]=ImageEnhance.Sharpness(video_rgb[i]).enhance(sharp_intensity)
    return imgs,video_rgb
