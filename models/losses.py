import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np
from torch import autograd
import torchvision.models as models
import util.util as util
from models.vgg import Vgg19
from torch.autograd import Function
from models.CX import CX_loss

###############################################################################
# Functions
###############################################################################
def compute_gradient(img):
    gradx=img[...,1:,:]-img[...,:-1,:]
    grady=img[...,1:]-img[...,:-1]
    return gradx,grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target) 
        
        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        device = next(self.vgg.parameters()).device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        
        return loss


class CXLoss(VGGLoss):
    # Contextual Loss from
    # https://arxiv.org/abs/1803.02077
    def __init__(self, vgg=None, weights=None, indices=None, criterions=None):        
        super(CXLoss, self).__init__(vgg, weights, indices)
        self.criterions = criterions or [CX_loss] * (len(weights))
    
    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterions[i](x_vgg[i], y_vgg[i].detach())
        
        loss = loss[0] if loss.dim() == 1 else loss
        return loss


def _sobel_kernels(device):
    """Return Sobel gradient kernels G_x and G_y as fixed Conv2d layers."""
    Gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    Gy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    return Gx, Gy


def _exclusion_loss_at_scale(predict, input, Gx, Gy):
    """Compute exclusion loss at a single scale between T_hat and R_derived=I-T_hat."""
    R_derived = input - predict
    groups = predict.size(1)

    grad_Tx = F.conv2d(predict, Gx, padding=1, groups=groups)
    grad_Ty = F.conv2d(predict, Gy, padding=1, groups=groups)
    grad_Rx = F.conv2d(R_derived, Gx, padding=1, groups=groups)
    grad_Ry = F.conv2d(R_derived, Gy, padding=1, groups=groups)

    grad_T_norm = torch.sqrt(grad_Tx ** 2 + grad_Ty ** 2 + 1e-6)
    grad_R_norm = torch.sqrt(grad_Rx ** 2 + grad_Ry ** 2 + 1e-6)

    grad_T_norm = grad_T_norm / (grad_T_norm.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)
    grad_R_norm = grad_R_norm / (grad_R_norm.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)

    return (grad_T_norm * grad_R_norm).mean()


class ExclusionLoss(nn.Module):
    """
    Gradient-domain exclusion loss [Zhang et al., CVPR 2018].
    Penalizes overlapping edges between T_hat and R_derived = I - T_hat.
    Operates at 3 scales: 1x, 0.5x, 0.25x.
    """
    def __init__(self, num_scales=3):
        super(ExclusionLoss, self).__init__()
        self.num_scales = num_scales

    def forward(self, predict, input):
        device = predict.device
        Gx, Gy = _sobel_kernels(device)

        if predict.size(1) == 3:
            Gx = Gx.repeat(3, 1, 1, 1)
            Gy = Gy.repeat(3, 1, 1, 1)
        else:
            Gx = Gx.repeat(predict.size(1), 1, 1, 1)
            Gy = Gy.repeat(predict.size(1), 1, 1, 1)

        loss = 0
        for i in range(self.num_scales):
            scale = 0.5 ** i
            if i > 0:
                pooled_pred = F.avg_pool2d(predict, kernel_size=2, stride=2)
                pooled_input = F.avg_pool2d(input, kernel_size=2, stride=2)
            else:
                pooled_pred = predict
                pooled_input = input
            loss += _exclusion_loss_at_scale(pooled_pred, pooled_input, Gx, Gy)

        return loss / self.num_scales


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss() # absorb sigmoid into BCELoss

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                target_tensor = self.get_target_tensor(input_i, target_is_real)
                loss += self.loss(input_i, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


class DiscLoss():
    def name(self):
        return 'SGAN'

    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA=None, fakeB=None, realB=None):
        pred_fake = None
        pred_real = None
        loss_D_fake = 0
        loss_D_real = 0
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero

        if fakeB is not None:
            pred_fake = net.forward(fakeB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, 0)

        # Real
        if realB is not None:
            pred_real = net.forward(realB)
            loss_D_real = self.criterionGAN(pred_real, 1)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, pred_fake, pred_real


class DiscLossR(DiscLoss):
    # RSGAN from 
    # https://arxiv.org/abs/1807.00734        
    def name(self):
        return 'RSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake - pred_real, 1)

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())

        loss_D = self.criterionGAN(pred_real - pred_fake, 1) # BCE_stable loss
        return loss_D, pred_fake, pred_real


class DiscLossRa(DiscLoss):
    # RaSGAN from 
    # https://arxiv.org/abs/1807.00734    
    def name(self):
        return 'RaSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)

        loss_G = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 0)
        loss_G += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 1)
        return loss_G * 0.5

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())
        
        loss_D = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 1)
        loss_D += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 0)
        return loss_D * 0.5, pred_fake, pred_real


def init_loss(opt, tensor):
    disc_loss = None
    content_loss = None

    loss_dic = {}

    pixel_loss = ContentLoss()
    pixel_loss.initialize(MultipleLoss([nn.MSELoss(), GradientLoss()], [0.2,0.4]))

    loss_dic['t_pixel'] = pixel_loss
    loss_dic['r_pixel'] = pixel_loss

    if opt.lambda_gan > 0:
        if opt.gan_type == 'sgan' or opt.gan_type == 'gan':
            disc_loss = DiscLoss()
        elif opt.gan_type == 'rsgan':
            disc_loss = DiscLossR()
        elif opt.gan_type == 'rasgan':
            disc_loss = DiscLossRa()
        else:
            raise ValueError("GAN [%s] not recognized." % opt.gan_type)

        disc_loss.initialize(opt, tensor)
        loss_dic['gan'] = disc_loss

    return loss_dic
