import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
from PIL import Image
from os.path import join
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch

from .errnet_model import ERRNetBase, tensor2im, _torch_load_compat, EdgeMap
from .arch.dual_stream import ResidualModule


class DualERRNetModel(ERRNetBase):
    def name(self):
        return 'dual_errnet'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cpu")

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()
        if hasattr(self, 'residual_module'):
            self.residual_module.eval()

    def _train(self):
        self.net_i.train()
        if hasattr(self, 'residual_module'):
            self.residual_module.train()

    def initialize(self, opt):
        ERRNetBase.initialize(self, opt)
        self.device = torch.device(
            "cuda:%d" % self.gpu_ids[0] if len(self.gpu_ids) > 0 else "cpu")

        in_channels = 3
        self.vgg = None

        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472

        self.net_i = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
        networks.init_weights(self.net_i, init_type=opt.init_type)

        self.residual_module = ResidualModule().to(self.device)

        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)

            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            rvggloss = losses.ContentLoss()
            rvggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['r_vgg'] = rvggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(
                    self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(
                    self.vgg, weights=[0.1, 0.1, 0.1],
                    indices=[8, 13, 22]))
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

            # Discard: CX on R is never used — real datasets have no true target_r

            self.loss_residual = losses.ResidualConsistencyLoss()

            # define discriminator
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                list(self.net_i.parameters()) +
                list(self.residual_module.parameters()),
                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)

        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = \
            self.loss_dic['gan'].get_loss(
                self.netD, self.input, self.output_t, self.target_t)

        (self.loss_D * self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False

        self.loss_G = 0
        self.loss_CX = None
        self.loss_t_pixel = None
        self.loss_t_vgg = None
        self.loss_r_pixel = None
        self.loss_r_vgg = None
        self.loss_residual_val = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_t, self.target_t)
            self.loss_G += self.loss_G_GAN * self.opt.lambda_gan

        if self.aligned:
            # T losses (always applicable for aligned data)
            self.loss_t_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_t, self.target_t)
            self.loss_t_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_t, self.target_t)
            self.loss_G += self.loss_t_pixel + \
                self.loss_t_vgg * self.opt.lambda_vgg

            # R losses only on synthetic data where target_r is real
            if self.issyn:
                self.loss_r_pixel = self.loss_dic['r_pixel'].get_loss(
                    self.output_r, self.target_r)
                self.loss_r_vgg = self.loss_dic['r_vgg'].get_loss(
                    self.output_r, self.target_r)
                self.loss_G += self.loss_r_pixel + \
                    self.loss_r_vgg * self.opt.lambda_vgg
        else:
            # CX on T only; skip CX on R since target_r is fake for all real data
            self.loss_CX = self.loss_dic['t_cx'].get_loss(
                self.output_t, self.target_t)
            self.loss_G += self.loss_CX

        # residual consistency loss (always computed, self-supervised)
        residual = self.residual_module(self.output_t, self.output_r)
        lambda_residual = getattr(self.opt, 'lambda_residual', 0.5)
        self.loss_residual_val = self.loss_residual(
            self.input, self.output_t, self.output_r, residual)
        self.loss_G += self.loss_residual_val * lambda_residual

        self.loss_G.backward()

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

        output_t, output_r = self.net_i(input_i)

        self.output_t = output_t
        self.output_r = output_r

        return output_t, output_r

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
        if self.loss_t_pixel is not None:
            ret_errors['TPixel'] = self.loss_t_pixel.item()
        if self.loss_t_vgg is not None:
            ret_errors['TVGG'] = self.loss_t_vgg.item()
        if self.loss_r_pixel is not None:
            ret_errors['RPixel'] = self.loss_r_pixel.item()
        if self.loss_r_vgg is not None:
            ret_errors['RVGG'] = self.loss_r_vgg.item()
        if self.loss_residual_val is not None:
            ret_errors['Residual'] = self.loss_residual_val.item()
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()
        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_t'] = tensor2im(self.output_t).astype(np.uint8)
        ret_visuals['output_r'] = tensor2im(self.output_r).astype(np.uint8)
        ret_visuals['target_t'] = tensor2im(self.target_t).astype(np.uint8)
        if self.target_r is not None:
            ret_visuals['target_r'] = tensor2im(self.target_r).astype(np.uint8)
        ret_visuals['residual'] = tensor2im(
            (self.input - self.output_t)).astype(np.uint8)
        return ret_visuals

    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            self.forward()

            output_t = tensor2im(self.output_t)
            target = tensor2im(self.target_t)

            if self.aligned:
                h = min(output_t.shape[0], target.shape[0])
                w = min(output_t.shape[1], target.shape[1])
                res = index.quality_assess(
                    output_t[:h, :w], target[:h, :w])
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(
                        os.path.basename(self.data_name[0]))[0]
                    if not os.path.exists(join(savedir, name)):
                        os.makedirs(join(savedir, name))
                    if suffix is not None:
                        Image.fromarray(output_t.astype(np.uint8)).save(
                            join(savedir, name,
                                 '{}_{}.png'.format(self.opt.name, suffix)))
                    else:
                        Image.fromarray(output_t.astype(np.uint8)).save(
                            join(savedir, name,
                                 '{}.png'.format(self.opt.name)))
                    Image.fromarray(target.astype(np.uint8)).save(
                        join(savedir, name, 't_label.png'))
                    Image.fromarray(
                        tensor2im(self.input).astype(np.uint8)).save(
                        join(savedir, name, 'm_input.png'))
                    # Also save reflection output
                    output_r = tensor2im(self.output_r)
                    Image.fromarray(output_r.astype(np.uint8)).save(
                        join(savedir, name, 'r_output.png'))
                else:
                    if not os.path.exists(join(savedir,
                                               'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(
                        join(savedir, 'transmission_layer',
                             str(self._count) + '.png'))
                    Image.fromarray(
                        tensor2im(self.input).astype(np.uint8)).save(
                        join(savedir, 'blended',
                             str(self._count) + '.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(
                os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))

            if os.path.exists(join(savedir, name,
                                   '{}.png'.format(self.opt.name))):
                return

        with torch.no_grad():
            self.forward()
            output_t = tensor2im(self.output_t)
            output_r = tensor2im(self.output_r)

            if self.data_name is not None and savedir is not None:
                Image.fromarray(output_t.astype(np.uint8)).save(
                    join(savedir, name, '{}.png'.format(self.opt.name)))
                Image.fromarray(output_r.astype(np.uint8)).save(
                    join(savedir, name, 'r_output.png'))
                Image.fromarray(
                    tensor2im(self.input).astype(np.uint8)).save(
                    join(savedir, name, 'm_input.png'))

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
            if 'residual' in state_dict:
                model.residual_module.load_state_dict(
                    state_dict['residual'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            state_dict = _torch_load_compat(
                icnn_path, map_location=torch.device('cpu'))
            pretrained_weights = state_dict['icnn']

            # Map old deconv3 (DRNet output head) -> head_T
            mapped_weights = {}
            for k, v in pretrained_weights.items():
                if k.startswith('deconv3'):
                    mapped_weights[k.replace('deconv3', 'head_T')] = v
                else:
                    mapped_weights[k] = v

            model.net_i.load_state_dict(mapped_weights, strict=False)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']

        if model.isTrain:
            if state_dict and 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])

        print('Resume from epoch %d, iteration %d' %
              (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'residual': self.residual_module.state_dict(),
            'opt_g': self.optimizer_G.state_dict(),
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict
