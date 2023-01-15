from IPython.display import Image as GIF
import imageio
import time
import cv2
from glob import glob
import os
import random
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

def generate_gif(directory_path, img_name,experiment_name,out_path):
    images = []
    keyword = experiment_name + '_' + img_name

    for filename in sorted(glob(directory_path+'/*')):       
        
        if filename.endswith(".png") and (keyword is None or keyword in filename):
            img_path = os.path.join(directory_path, filename)
            # print(img_path)
            images.append(imageio.imread(img_path))

    imageio.mimsave(
            os.path.join(out_path, 'anim_{}_{}.gif'.format(experiment_name,img_name)), images)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

 def pix_wise_accuracy(gt_image_path,translated_image_path,tolerance=3):
    gt_image =  cv2.imread(gt_image_path,0)
    translated_image = cv2.imread(translated_image_path,0)
    h,w = gt_image.shape
    total_pix = h * w
    t = 0
    f = 0
    # Recursively computing neighboring cells
    for i in range(len(translated_image)):
        for j in range(len(translated_image[0])):
            pix_val = translated_image[i][j]
            orig_val = gt_image[i][j]
            if (orig_val >= pix_val -150) and (orig_val <= pix_val + tolerance):
                t +=1
            else:
                f +=1
    return t / total_pix
