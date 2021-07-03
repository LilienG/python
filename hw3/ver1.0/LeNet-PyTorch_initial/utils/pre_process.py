#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/28 17:42
# @File     : pre_process.py

"""
import torchvision
from torchvision import transforms
from torchtoolbox.transform import Cutout


def normal_transform():
    normal = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return normal


# Todo: Add Pre_Process, adjust your parameters——task five
def data_augment_transform():
    data_augment = torchvision.transforms.Compose([
        # RandomCrop
        transforms.RandomCrop(size=30, padding=2),
        Cutout(0.5),
        # HorizontalFlip
        transforms.RandomHorizontalFlip(p=0.5),
        # VerticalFlip
        transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ToTensor()
    ])
    return data_augment
