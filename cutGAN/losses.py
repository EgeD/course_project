import torch
import functools
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from slot_attention import SlotAttention
from torchsummary import summary

class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):

        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = None


    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real) 
            loss = self.loss(prediction, target_tensor)

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class PatchNCELoss(nn.Module):
    def __init__(self,batch_size,nce_includes_all_negatives_from_minibatch=False,nce_T=0.07):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.batch_size = batch_size
        self.nce_T = nce_T # Temperature for NCE loss

    def forward(self,feature_q, feature_k):
        num_patches = feature_q.shape[0]
        dim = feature_q.shape[1]
        feature_k = feature_k.detach()

        #Positive Logits

        pos_logits = torch.bmm(
            feature_q.view(num_patches, 1, -1), feature_k.view(num_patches, -1, 1))

        pos_logits = pos_logits.view(num_patches,1)

        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        feature_q = feature_q.view(batch_dim_for_bmm, -1, dim)
        feature_k = feature_k.view(batch_dim_for_bmm, -1, dim)
        num_patches = feature_q.size(1)

        neg_logits_for_batch = torch.bmm(feature_q, feature_k.transpose(2, 1))
        
        diagonal = torch.eye(num_patches, device=feature_q.device, dtype=self.mask_dtype)[None, :, :]
        neg_logits_for_batch.masked_fill_(diagonal, -10.0)
        neg_logits = neg_logits_for_batch.view(-1, num_patches)

        out = torch.cat((pos_logits, neg_logits), dim=1) / self.nce_T

        loss = self.cross_entropy(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feature_q.device))

        return loss

