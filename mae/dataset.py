import os
import cv2
import tifffile as tf
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn 
import torchvision.transforms as transforms
import albumentations as A
import math
import random

def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num

def apply_transform(image, mask):
    strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]
    level = 5
    transform = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                    crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 2 * level), 'y': (0, 0)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),  # x
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 0), 'y': (0, 2 * level)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level)
    ])
    employ = random.choice(strategy)
    level, shape = random.sample(transform[:6], employ[0]), random.sample(transform[6:], employ[1])
    img_transform = A.Compose([*level, *shape])
    random.shuffle(img_transform.transforms)
    transformed = img_transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']

class Vessile(Dataset):
    def __init__(self, dir, split = 'train'):
        self.dir = dir 
        self.file_list = os.listdir(dir)
        self.resize_size = [224,224]
        self.split = split
        self.mean = 0.094
        self.std = 0.049

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.dir,self.file_list[idx])
        image = tf.imread(path).astype(np.float32) 
        image = np.expand_dims(image, axis=0) 
        image /= 65535. # 16bit
        image = image*255.
        image = image.astype(np.uint8).transpose(1,2,0)
        
        image = cv2.resize(image, self.resize_size)
        label = np.zeros_like(image)
        
        if self.split == 'train':
            image,label = apply_transform(image,label)
        image = image/255.
        image = (image - self.mean) / self.std
        image = np.expand_dims(image, axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label)
        name = path.split('/')[-1]
        return image.float(), label.float(), name
