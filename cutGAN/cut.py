import torch
import torch.nn as nn
import itertools
import os
from image_pool import ImagePool
from discriminator import Discriminator, PatchSampleF
from generator import Generator
from losses import GANLoss, PatchNCELoss
from torch.optim import lr_scheduler
from collections import OrderedDict

class CUT(nn.Module):
    def __init__(self,input_dim,output_dim, n_epochs,n_epochs_decay,batch_size=1,norm_layer=nn.InstanceNorm2d, n_filter=64, lambda_GAN=1.0, lambda_NCE=1.0, nce_idt=False, 
                    use_dropout=False, use_bias=False, padding_type='reflect', gan_mode='lsgan', beta1=0.5, beta2=0.999, lr=0.0002,
                    nce_layers=[0,4,8,12,16], nce_includes_all_negatives_from_minibatch=False, num_patches=256,lr_decay_iters=50,lr_policy='linear'):

        super().__init__()
        if batch_size > 1:
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = norm_layer

        self.n_filter = n_filter
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.nce_idt = nce_idt
        self.nce_layers = nce_layers
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.num_patches = num_patches
        self.batch_size = batch_size

        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.optimizers = []
        visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'F', 'D']

        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.lr_decay_iters =  lr_decay_iters
        self.lr_policy = lr_policy
        self.epoch_count = 1

        # self.lambda_idt = 0.5
        # self.lambda_A = 10.0
        # self.lambda_B = 10.0
        self.direction = 'AtoB'
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr

        self.device = torch.device("cuda:0")
        # define networks (both Generators and discriminators)

        self.netG = Generator(input_dim,output_dim,n_filter=n_filter,norm_layer=norm_layer,use_dropout=use_dropout,use_bias=use_bias,padding_type=padding_type)
        self.netF = PatchSampleF(self.num_patches)
        self.netD = Discriminator(input_dim,n_filter=n_filter,norm_layer=norm_layer)


        # define loss functions
        self.criterionGAN = GANLoss(gan_mode=gan_mode).to(self.device)  # define GAN loss.
        self.criterionNCE = []

        for _ in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(self.batch_size).to(self.device))

        self.criterionIdt = nn.L1Loss().to(self.device)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)


    def init_according_to_data(self,data):

        self.set_input(data)
        self.forward()

        self.compute_D_loss().backward()                  
        self.compute_G_loss().backward()    

        self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        self.optimizers.append(self.optimizer_F)


    def set_input(self, input):
        AtoB = self.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B), dim=0)
        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]

    def compute_D_loss(self):
        fake = self.fake_B.detach()
        #Fake Image 
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        #Real Image
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        return self.loss_D

    def compute_G_loss(self):
        fake = self.fake_B
        #Fake Image 
        pred_fake = self.netD(fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.lambda_GAN

        self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        
        self.loss_G = self.loss_G_GAN + self.loss_NCE

        return self.loss_G


    def calculate_NCE_loss(self,source,target):
        num_layers = len(self.nce_layers)
        features_q = self.netG(target,self.nce_layers,encode=True)

        features_k = self.netG(source, self.nce_layers, encode=True)
        features_k_pool, sample_ids = self.netF(features_k, self.num_patches)
        features_q_pool, _ = self.netF(features_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0

        for feature_q, feature_k, criterion, nce_layer in zip(features_q_pool, features_k_pool, self.criterionNCE, self.nce_layers):
            loss = criterion(feature_q, feature_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / num_layers

    def optimize_parameters(self):
        # forward
        self.forward()      
        
        self.set_requires_grad(self.netD, True)  
        self.optimizer_D.zero_grad()   # set D's gradients to zero
        self.loss_D = self.compute_D_loss() 
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()  

        self.loss_G = self.compute_G_loss() 
        self.loss_G.backward()

        self.optimizer_G.step()
        self.optimizer_F.step()



    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def setup(self):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.schedulers = [self.get_scheduler(optimizer, self.lr_policy) for optimizer in self.optimizers]


    def get_scheduler(self,optimizer, lr_policy):
        """Return a learning rate scheduler

        Parameters:
            optimizer          -- the optimizer of the network
            opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
                                opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

        For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
        and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
        For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
        See https://pytorch.org/docs/stable/optim.html for more details.
        """
        if lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + self.epoch_count - self.n_epochs) / float(self.n_epochs_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_iters, gamma=0.1)
        elif lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=0)
        return scheduler

    def save_networks(self, epoch,experiment_name,save_dir):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s_net_%s.pth' % (experiment_name,epoch, name)
                # save_dir = '/kuacc/users/edincer16/comp547/course_project/saved_models'
                save_path = os.path.join(save_dir, save_filename)
                
                net = getattr(self, 'net' + name)
                torch.save(net.cpu().state_dict(), save_path)


    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret