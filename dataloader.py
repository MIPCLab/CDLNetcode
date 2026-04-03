from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import cv2 as cv
import PIL.ImageDraw
import torch.nn as nn
from torchvision.models import resnet50
import argparse
import torch.optim as optim
# from .randaugment import TransformFixMatchMedium
import os
import torch
import torchvision
from loss import wavelet_filter,non_local_means_filter,mixup_criterion,mixup_data,check_and_correct_image

PARAMETER_MAX = 10

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)
def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX
def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int64)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))



def new_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.1, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.1, 0),
            (ShearY, 0.1, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.05, 0),
            (TranslateY, 0.05, 0)
            ]
    return augs


class RandAugmentwogeo(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = new_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        size = img.size[0]
        img = CutoutAbs(img, int(size * 0.15))
        return img

class TransformFixMatchMedium(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=256,
            #                       padding=int(256*0.125),
            #                       padding_mode='reflect')])
            transforms.Resize([256,256])])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize([256,256]),
            # transforms.RandomCrop(size=256,
            #                       padding=int(256*0.125),
            #                       padding_mode='reflect'),
            RandAugmentwogeo(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.origen = transforms.Compose([
            transforms.Resize([256,256]),
            # transforms.CenterCrop(64),,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.origen(x)
class FGSC():

    def __init__(self, root_dir, mode):

        self.root_dir = root_dir
        self.mode = mode
    
        self.transform_test = transforms.Compose([
            transforms.Resize([256,256]),
            # transforms.CenterCrop(64),,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        self.transform_train = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        self.transform_fixmatch = TransformFixMatchMedium(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.train_imgs = []
        # for i in range (1,19):
        # self.train_dir = root_dir + '/train/'
        # for subdir in os.listdir(self.train_dir):
        #     subdir_path = os.path.join(self.train_dir, subdir)
        #     if os.path.isdir(subdir_path):
        #         train_imgs = os.listdir(subdir_path)
        #         for img in train_imgs:
        #             self.train_imgs.append([subdir_path+'/'+img, int(subdir)])
        
        # self.test_imgs = []
        # self.test_dir = root_dir +"/test/" 
        # for subdir in os.listdir(self.test_dir):
        #     subdir_path = os.path.join(self.test_dir, subdir)
        #     if os.path.isdir(subdir_path):
        #         test_imgs = os.listdir(subdir_path)
        #         for img in test_imgs:
        #             self.test_imgs.append([subdir_path+'/'+img,int(subdir)])
        self.train_dir = root_dir + '/train/'
        for subdir in os.listdir(self.train_dir):
            subdir_path = os.path.join(self.train_dir, subdir)
            classname = int(subdir.split('.')[0])-1
            if os.path.isdir(subdir_path):
                train_imgs = os.listdir(subdir_path)
                for img in train_imgs:
                    self.train_imgs.append([subdir_path+'/'+img, classname])
        
        self.test_imgs = []
        self.test_dir = root_dir +"/test/" 
        for subdir in os.listdir(self.test_dir):
            subdir_path = os.path.join(self.test_dir, subdir)
            classname = int(subdir.split('.')[0])-1
            if os.path.isdir(subdir_path):
                test_imgs = os.listdir(subdir_path)
                for img in test_imgs:
                    self.test_imgs.append([subdir_path+'/'+img,classname])

    def __getitem__(self, index):
        if self.mode == 'train':
            img_id, target = self.train_imgs[index]
            img_path =  img_id
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target,img_path.split('/')[-1]

        elif self.mode == 'test':
            ind = index
            img_id, target = self.test_imgs[index]
            img_path =  img_id
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':#
            return len(self.test_imgs)
        elif self.mode == 'train' or self.mode == 'train_index' or self.mode == 'train_single':
            return len(self.train_imgs)
        
def get_dataloader( batch_size=16, num_workers=0, shuffle=True):

    

    # train_data =  FGSC('FGSC-23',  mode='train')
    # test_data =  FGSC('FGSC-23',  mode='test')
    train_data =  FGSC('FGSCR-test',  mode='train')
    test_data =  FGSC('FGSCR-test',  mode='test')
    
    train_loader = DataLoader(
        train_data, shuffle=True, num_workers=num_workers, batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(
        test_data, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    
    
    return train_loader, test_loader