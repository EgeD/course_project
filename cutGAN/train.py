import time
import cv2
import glob
import os
import random
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from cut import CUT
from PIL import Image
from torch.utils.data import Dataset

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


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + "/*.*"))
        self.a_path = os.path.join(root, f"{mode}A")
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B") + "/*.*"))
        self.b_path = os.path.join(root, f"{mode}B")
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        
        return {"A": item_A, "B": item_B,"A_paths":self.a_path,"B_paths":self.b_path}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__ == '__main__':
    # Model initializations
    n_epochs = 200
    n_epochs_decay = n_epochs
    batch_size = 1
    model = CUT(input_dim=3,output_dim=3,n_epochs=n_epochs,lr_decay_iters=n_epochs_decay,
    n_epochs_decay=n_epochs_decay,batch_size=batch_size,num_patches=256,
    lambda_GAN=4.0, lambda_NCE=4.0,gan_mode='lsgan').cuda()
    experiment_name = 'cut_baseline'
    data_folder = '/kuacc/users/edincer16/Comp541_fall22/course_project/cutGAN/horse2zebra'

    transform = transforms.Compose([
    transforms.Resize(286,Image.BICUBIC),
    transforms.RandomCrop(256),
    transforms.ToTensor()
    ])
    dataset = ImageDataset(data_folder,transform=transform,unaligned=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Data initializations 

    test_img_path = '/kuacc/users/edincer16/Comp541_fall22/course_project/cutGAN/horse2zebra/valA/*.jpg'
    output_path = '/kuacc/users/edincer16/Comp541_fall22/course_project/cutGAN/model_results/baseline_results/'

   # transform = transforms.ToTensor()


    dataset_size = len(dataset)    # get the number of images in the dataset.

    # saved_model_path = '/kuacc/users/edincer16/comp547/course_project/CelebA/beard_ckpt'

    saved_model_path = '/kuacc/users/edincer16/Comp541_fall22/course_project/cutGAN/model_results/baseline_weights'

    logger = open(f'/kuacc/users/edincer16/Comp541_fall22/course_project/cutGAN/{experiment_name}.txt','w')

    print('The number of training images = %d' % dataset_size)
    total_iters = 0                # the total number of training iterations

    for epoch in tqdm(range(0, n_epochs + n_epochs_decay + 1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        
        for i, data in enumerate(dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            
            if epoch == 0 and i==0:
                model.init_according_to_data(data)
                model.setup()
            
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            iter_data_time = time.time()


        for img in glob.glob(test_img_path):
            # img_tmp = cv2.imread(img)
            img_tmp = Image.open(img).convert('RGB')
            # img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
            image = transform(img_tmp).unsqueeze(0)
            generated_image = model.netG(image.cuda())
            image_name = img.split('/')[-1].split('.')[0] +'_'+str(epoch)+'.png'
            gen_img = tensor2im(generated_image)
            cv2.imwrite(output_path+f'{experiment_name}_'+image_name,cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR))

        current_losses = model.get_current_losses()
        log_write = f'Epoch: {epoch}, The losses are '+ str(current_losses) + '\n'
        logger.write(log_write)
        if epoch % 4 == 0:
            model.save_networks(epoch,experiment_name,saved_model_path)
            model.cuda() # sending model back to gpu

        model.update_learning_rate()  

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, n_epochs + n_epochs_decay, time.time() - epoch_start_time))
    logger.close()
