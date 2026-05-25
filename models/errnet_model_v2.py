"""ERRNet V2 model with Transformer backbone and pluggable frequency enhancement."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from collections import OrderedDict
from os.path import join
from PIL import Image

import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch
from models.errnet_model import ERRNetBase, tensor2im, _flag_enabled, _torch_load_compat, EdgeMap

from models.arch.errnet_transformer import ERRNetTransformer
from models.arch.frequency import FrequencyEnhancementModule


class ERRNetModelV2(ERRNetBase):
    """Improved ERRNet model using Transformer backbone and optional frequency module."""

    def name(self):
        return 'errnet_v2'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cpu")

    def print_network(self):
        print('--------------------- Model V2 ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()

    def _train(self):
        self.net_i.train()

    def initialize(self, opt):
        ERRNetBase.initialize(self, opt)
        self.device = torch.device(
            "cuda:%d" % self.gpu_ids[0] if len(self.gpu_ids) > 0 else "cpu"
        )

        in_channels = 3
        self.vgg = None

        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472

        self.net_i = ERRNetTransformer(
            in_channels, 3,
            n_feats=256,
            n_blocks=opt.n_transformer_blocks,
            num_heads=opt.n_heads,
            res_scale=0.1,
            bottom_kernel_size=1,
            pyramid=True,
        ).to(self.device)

        networks.init_weights(self.net_i, init_type=opt.init_type)
        self.edge_map = EdgeMap(scale=1).to(self.device)

        self.freq_module = FrequencyEnhancementModule(
            256, enabled=opt.use_frequency_module
        ).to(self.device)

        if self.isTrain:
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(
                    self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(
                    self.vgg, weights=[0.1, 0.1, 0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(
                    self.vgg, weights=[0.1, 0.1, 0.1, 0.1],
                    indices=[8, 13, 22, 31],
                    criterions=[losses.CX_loss] * 3 + [nn.L1Loss()]))
            else:
                raise NotImplementedError
            self.loss_dic['t_cx'] = cxloss

            self.fft_loss = losses.FFTLoss().to(self.device)

            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            self.optimizer_G = torch.optim.Adam(
                self.net_i.parameters(), lr=opt.lr,
                betas=(0.9, 0.999), weight_decay=opt.wd)
            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)

        if opt.no_verbose is False:
            self.print_network()

    def forward(self):
        input_i = self.input

        if self.vgg is not None:
            hypercolumn = self.vgg(self.input)
            _, C, H, W = self.input.shape
            hypercolumn = [
                F.interpolate(feature.detach(), size=(H, W),
                              mode='bilinear', align_corners=False)
                for feature in hypercolumn
            ]
            input_i = [input_i]
            input_i.extend(hypercolumn)
            input_i = torch.cat(input_i, dim=1)

        features = self.net_i.forward_features(input_i)
        features = self.freq_module(features)
        output_i = self.net_i.forward_head(features)

        self.output_i = output_i
        return output_i

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D * self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False

        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None
        self.loss_fft = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t)
            self.loss_G += self.loss_G_GAN * self.opt.lambda_gan

        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)
            self.loss_G += self.loss_icnn_pixel + self.loss_icnn_vgg * self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(
                self.output_i, self.target_t)
            self.loss_G += self.loss_CX

        if self.opt.lambda_fft > 0:
            self.loss_fft = self.fft_loss(self.output_i, self.target_t)
            self.loss_G += self.loss_fft * self.opt.lambda_fft

        self.loss_G.backward()

    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()
        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()
        if self.loss_fft is not None:
            ret_errors['FFT'] = self.loss_fft.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)
        return ret_visuals

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None

        if icnn_path is None:
            model_path = util.get_model_list(
                model.save_dir, model.name(), epoch=resume_epoch)
            state_dict = _torch_load_compat(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            model.net_i.load_state_dict(state_dict['icnn'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            state_dict = _torch_load_compat(icnn_path, map_location=torch.device('cpu'))
            model.net_i.load_state_dict(state_dict['icnn'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']

        if model.isTrain:
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])

        print('Resume from epoch %d, iteration %d' %
              (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(),
            'epoch': self.epoch,
            'iterations': self.iterations,
        }
        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })
        return state_dict
