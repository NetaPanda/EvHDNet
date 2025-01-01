import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
import numpy as np
from .base_model import BaseModel
import model as Model
import copy
logger = logging.getLogger('base')



class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models

        # ddpm denoiser (used only for training)
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.AdamW(
                optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.9,0.9), weight_decay=1e-4)
            self.log_dict = OrderedDict()

        # event integration net (branch)
        int_net = Model.create_int_net(opt)
        if opt['gpu_ids'] and opt['distributed']:
            assert torch.cuda.is_available()
            int_net = nn.DataParallel(int_net)
        self.int_net = self.set_device(int_net)
        if self.opt['phase'] == 'train':
            self.int_net.train()
        optim_params = list(self.int_net.parameters())
        self.optI = torch.optim.AdamW(
                    optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.9,0.9), weight_decay=1e-4)
        # spectral coefficient prediction net (branch)
        ddpm_init_net = Model.create_ddpm_init_net(opt)
        if opt['gpu_ids'] and opt['distributed']:
            assert torch.cuda.is_available()
            ddpm_init_net = nn.DataParallel(ddpm_init_net)
        self.ddpm_init_net = self.set_device(ddpm_init_net)
        if self.opt['phase'] == 'train':
            self.ddpm_init_net.train()
        optim_params = list(self.ddpm_init_net.parameters())
        self.optD = torch.optim.AdamW(
                    optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.9,0.9), weight_decay=1e-4)

        # light-weight encoder in DNA
        standalone_encoder = Model.create_standalone_encoder(opt)
        if opt['gpu_ids'] and opt['distributed']:
            assert torch.cuda.is_available()
            standalone_encoder = nn.DataParallel(standalone_encoder)
        self.standalone_encoder = self.set_device(standalone_encoder)
        if self.opt['phase'] == 'train':
            self.standalone_encoder.train()
        optim_params = list(self.standalone_encoder.parameters())
        self.optE = torch.optim.AdamW(
                    optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.9,0.9), weight_decay=1e-4)


        self.load_network()
        self.print_network()


        if self.opt['phase'] == 'train':
            self.warmup_iter = 0
            self.optimizers = [self.optG, self.optI, self.optD, self.optE]
            self.schedulers = []
            self.setup_schedulers()


    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters_ori(self):
        self.optG.zero_grad()
        self.optE.zero_grad()
        self.optI.zero_grad()
        self.optD.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['hsi_gt'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()

        self.optG.step()
        self.optE.step()
        self.optI.step()
        self.optD.step()

        self.log_dict['l_pix'] = l_pix.item()
        lr = self.get_current_learning_rate()
        self.log_dict['lr'] = lr[0]
        self.update_learning_rate(100) # 100 is dummy value, just > 1 will ok


    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()

    def test_deblur(self, continous=False):
        self.netG.eval()
        self.int_net.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.estimated_W = self.netG.module.deblur_w_estimation(
                    self.data, continous)
            else:
                self.estimated_W = self.netG.deblur_w_estimation(
                    self.data, continous)
        self.netG.train()
        self.int_net.train()



    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def get_current_visuals_deblur(self, logI, logW_init, logW_residual=None):
        logI = logI.detach().float().cpu() # I is the D in paper, which is the event double integral
        logW_init = logW_init.detach().float().cpu()
        out_dict = OrderedDict()
        if logW_residual is None:
            logW_residual = self.estimated_W.detach().float().cpu()
        logW_residual = logW_residual.detach().float().cpu()
        logW = logW_init + logW_residual
        out_dict['W'] = torch.exp(logW + 0)
        W = out_dict['W']
        out_dict['B'] = self.data['hsi_blur'].detach().float().cpu()
        B = out_dict['B']
        logB = torch.log(B+0)
        out_dict['I'] = torch.exp(logI.detach().float().cpu() + 0)
        I = out_dict['I']
        out_dict['L'] = torch.exp(logB - logW - logI)  # recovered sharp frame
        out_dict['W_init'] = torch.exp(logW_init + 0)
        out_dict['W_res'] = torch.exp(logW_residual + 0)
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def save_network_deblur(self, epoch, iter_step, integrate_net=None):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        int_net_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_int.pth'.format(iter_step, epoch))
        ddpm_init_net_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_din.pth'.format(iter_step, epoch))
        stdalone_encoder_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_std.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # int
        network = self.int_net
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, int_net_path)
        # din
        network = self.ddpm_init_net
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, ddpm_init_net_path)
        # std
        network = self.standalone_encoder
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, stdalone_encoder_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            int_path = '{}_int.pth'.format(load_path)
            ddpm_init_path = '{}_din.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            stdalone_encoder_path = '{}_std.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # int
            network = self.int_net
            if isinstance(network, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                int_path), strict=(not self.opt['model']['finetune_norm']))
            # DDPM init
            network = self.ddpm_init_net
            if isinstance(network, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                ddpm_init_path), strict=(not self.opt['model']['finetune_norm']))
            # standalone encoder
            network = self.standalone_encoder
            if isinstance(network, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                stdalone_encoder_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']


    def get_model_opt_states(self):
        # gen
        all_state_dicts = []
        all_opt_states = []
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = copy.deepcopy(network.state_dict())
        all_state_dicts.append(state_dict)
        # int
        network = self.int_net
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        state_dict = copy.deepcopy(network.state_dict())
        all_state_dicts.append(state_dict)
        # din
        network = self.ddpm_init_net
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        state_dict = copy.deepcopy(network.state_dict())
        all_state_dicts.append(state_dict)
        # std
        network = self.standalone_encoder
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        state_dict = copy.deepcopy(network.state_dict())
        all_state_dicts.append(state_dict)
        # opt
        opt = self.optG
        state_dict = copy.deepcopy(opt.state_dict())
        all_opt_states.append(state_dict)

        opt = self.optI
        state_dict = copy.deepcopy(opt.state_dict())
        all_opt_states.append(state_dict)

        opt = self.optD
        state_dict = copy.deepcopy(opt.state_dict())
        all_opt_states.append(state_dict)

        opt = self.optE
        state_dict = copy.deepcopy(opt.state_dict())
        all_opt_states.append(state_dict)

        return all_state_dicts, all_opt_states


    def load_model_opt_states(self, all_state_dicts, all_opt_states):
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        network.load_state_dict(all_state_dicts[0], strict=True)
        # int
        network = self.int_net
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(all_state_dicts[1], strict=True)
        # DDPM init
        network = self.ddpm_init_net
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(all_state_dicts[2], strict=True)
        # standalone encoder
        network = self.standalone_encoder
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(all_state_dicts[3], strict=True)

        self.optG.load_state_dict(all_opt_states[0])
        self.optI.load_state_dict(all_opt_states[1])
        self.optD.load_state_dict(all_opt_states[2])
        self.optE.load_state_dict(all_opt_states[3])

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    
    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr



    def get_current_learning_rate(self):
        return [
            param_group['lr']
            for param_group in self.optimizers[0].param_groups
        ]



