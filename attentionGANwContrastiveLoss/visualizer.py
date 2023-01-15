import numpy as np
import os
import sys
import ntpath
import time
import torch
from collections import OrderedDict

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)


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
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, experiment_name,checkpoints_dir):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """

        self.name = experiment_name
        self.saved = False
        self.use_wandb = True
        self.current_epoch = 0

        self.wandb_run = wandb.init(project='ContrastiveAttention', name=experiment_name) if not wandb.run else wandb.run

        # create a logging file to store training losses
        path = os.path.join(checkpoints_dir, experiment_name)
        if os.path.exists(path):
            self.log_name = os.path.join(checkpoints_dir, experiment_name, 'loss_log.txt')
        else:
            os.makedirs(path)
            self.log_name = os.path.join(checkpoints_dir, experiment_name, 'loss_log.txt')


        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False


    def display_current_results(self, visuals, epoch):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
        """
        columns = [key for key, _ in visuals.items()]
        columns.insert(0,'epoch')
        result_table = wandb.Table(columns=columns)
        table_row = [epoch]
        ims_dict = {}
        for label, image in visuals.items():
            image_numpy = tensor2im(image)
            wandb_image = wandb.Image(image_numpy)
            table_row.append(wandb_image)
            ims_dict[label] = wandb_image
        self.wandb_run.log(ims_dict)
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            result_table.add_data(*table_row)
            self.wandb_run.log({"Result": result_table})



    def plot_current_losses(self, losses):

        self.wandb_run.log(losses)
