import torch
import sys,os
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage

def run():
    a=torch.randn([2,3])
    b=torch.randn(a.size())
    print(b)
    print(np.random.beta(1, 1))

def to_grey(img_path):
    img = Image.open(img_path)
    img.show()
    box = (200, 100, 1400, 1100)
    tmp_img = img.crop(box)
    grey_img = tmp_img.convert("L")
    grey_img.show()
    grey_img = grey_img.resize((256,256))
    grey_img.show()
    return grey_img


if __name__ == '__main__':
    img_path = '../Data/ROP/1480457/3a744179-097f-4a08-97c1-2320c8f6578f.4.png'
    to_grey(img_path)
    # img = Image.open(path)
    # data = np.asarray(img,dtype=np.float)
    # print(data.shape,data.dtype)
    # print(os.path.dirname(sys.path[0]))