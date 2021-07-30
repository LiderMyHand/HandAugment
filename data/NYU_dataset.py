import os
import cv2
import torch
import torch.utils.data
import numpy as np
import re
import scipy.io as scio
from .util import crop_image

img_w = 640
img_h = 480
fx = 588.03
fy = -587.07
ppx = 320
ppy = 240
cube_len = 150
channel = 1
joint_n = 21


def read_image(img_path):
    img_depth = cv2.imread(img_path, 2).astype(np.float32)
    return img_depth


class NYUTestDataset(torch.utils.data.Dataset):
    def __init__(self, center_list_path, img_base, crop_width=224, crop_height=224):
        lines = [line.split() for line in open(center_list_path, 'r').readlines()]
        self.path_list = [os.path.join(img_base, line[0]) for line in lines]
        self.center_list = [[float(x) for x in line[1:]] for line in lines]
        self.crop_width = crop_width
        self.crop_height = crop_height

    def __getitem__(self, index):
        img_path = self.path_list[index]
        img_depth = self._read_image(img_path)
        center = self.center_list[index]
        if center[2] == 0:
            data = torch.FloatTensor(channel, self.crop_height, self.crop_width)
            crop_param = np.array(center, dtype=np.float32)
            return data, crop_param
        img_crop = crop_image(img_depth, center, cube_len, fx, fy, self.crop_width, self.crop_height)
        data = torch.from_numpy(
            np.asarray(img_crop, dtype=np.float32).reshape(channel, self.crop_height, self.crop_width))
        crop_param = np.array(center, dtype=np.float32)
        return data, crop_param

    def __len__(self):
        return len(self.path_list)

    def _read_image(self, img_path):
        return read_image(img_path)

def prepare_centers_uvd_from_raw(centers='/HandAugment_method/HandAugment/cache/NYU/nyu_center_test.mat',
                        dst_path='/HandAugment_method/HandAugment/cache/NYU/test_center_uvd.txt'):
    
    
    lines  = scio.loadmat(centers)['centre_pixel'].astype(np.float32)

    filelist= [file for file in os.listdir('/HandAugment_method/HandAugment/dataset/NYU') 
               if file.endswith('.png') and re.search("^depth_2", file)]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    fo = open(dst_path, 'w')

    for bb, file in zip(lines, filelist):
        for i in bb:  
            i = [float(x) for x in i]
            center_str = ['\t' + '%.4f' % (x) for x in i]
            fo.write(file)
            fo.writelines(center_str)
            fo.write('\n') 






