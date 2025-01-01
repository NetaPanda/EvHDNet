import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
import scipy.io as sio
import time
from einops import rearrange, reduce, repeat
from pynvml import *
from augments import augment_sample
nvmlInit()


# crop patch and increase batch size by batch_multiplier times
def random_crop_patch(train_data, patch_size=128, batch_multiplier=16):
    if patch_size < 0:
        return train_data
    patch_data = {}
    blur = train_data['hsi_blur']
    H = blur.shape[2]
    W = blur.shape[3]
    for i in range(batch_multiplier):
        h_start = random.randint(0, H-patch_size)
        w_start = random.randint(0, W-patch_size)
        for key in train_data.keys():
            if key not in patch_data:
                patch_data[key] = []
            patch_data[key].append(train_data[key][:,:,h_start:h_start+patch_size, w_start:w_start+patch_size])
    for key in patch_data.keys():
        patch_data[key] = torch.cat(patch_data[key], dim=0)

    return patch_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/deblur.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or test(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)


    while True:
        h = nvmlDeviceGetHandleByIndex(opt['gpu_ids'][0])
        info = nvmlDeviceGetMemoryInfo(h)
        GB = 1024 * 1024 * 1024
        if info.free < 15 * GB:
            print("free     : %02f"%(info.free/GB))
            print('used     : %02f'%(info.used/GB))
            time.sleep(1)
        else:
            break


    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            train_set = Data.create_dataset_gopro_xima_event(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'test':
            test_set = Data.create_dataset_gopro_xima_event(dataset_opt, phase)
            test_loader = Data.create_dataloader(
                test_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    # event integration net
    int_net = diffusion.int_net
    # spectral coefficient calculation net
    ddpm_init_net = diffusion.ddpm_init_net
    # light weight encoder in DNA
    standalone_encoder = diffusion.standalone_encoder
    logger.info('Initial Model Finished')

    int_net.train()
    standalone_encoder.train()
    ddpm_init_net.train()
    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    patch_crops_per_img = opt['datasets']['train']['patch_crops_per_img']
    batch_multiplier = opt['datasets']['train']['batch_multiplier']
    patch_size = opt['datasets']['train']['patch_size']
    patch_crops_per_img_val = opt['datasets']['val']['patch_crops_per_img']
    patch_size_val = opt['datasets']['val']['patch_size']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        valid_count = 0
        all_count = 0
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += patch_crops_per_img
                if current_step > n_iter:
                    break
                # since data reading is slow, we load one batch of data and crop multiple patches from it for speed up
                # this issue can be solved by using advanced prefetchers
                for patch_idx in range(patch_crops_per_img):
                    train_data_patch = random_crop_patch(train_data, patch_size,batch_multiplier)
                    if random.random() < 0.5:
                        bs = train_data_patch['event_bin'].shape[0]
                        train_data_patch['event_bin'][0:bs,:,:,:] = torch.rot90(train_data_patch['event_bin'][0:bs,:,:,:], 1, [2,3])
                        train_data_patch['hsi_blur'][0:bs,:,:,:] = torch.rot90(train_data_patch['hsi_blur'][0:bs,:,:,:], 1, [2,3])
                        train_data_patch['hsi_gt'][0:bs,:,:,:] = torch.rot90(train_data_patch['hsi_gt'][0:bs,:,:,:], 1, [2,3])
                    train_data_patch = diffusion.set_device(train_data_patch)
                    train_data_patch['event_bin'], train_data_patch['hsi_blur'], train_data_patch['hsi_gt'] = augment_sample(train_data_patch['event_bin'], train_data_patch['hsi_blur'], train_data_patch['hsi_gt'])
                    event_bin = train_data_patch['event_bin']
                    hsi_blur = train_data_patch['hsi_blur'] # this is B
                    hsi_gt = train_data_patch['hsi_gt']     # this is L
                    event_tensor = train_data_patch['event_bin']
                    diffusion.feed_data(train_data_patch)
                    condition = torch.cat([event_tensor,hsi_blur],dim=1)
                    diffusion.data['condition'] = condition
                    # get the diffusion-guided noise aware features from light-weight encoder
                    salone_enc_feat_down,salone_enc_feat_mid = standalone_encoder(condition)
                    # detached ver
                    salone_enc_feat_down_detach = [f.detach() for f in salone_enc_feat_down]
                    salone_enc_feat_mid_detach = [f.detach() for f in salone_enc_feat_mid]

                    # calculate the event double integrate result D
                    int_res, int_enc_feat_down, int_enc_feat_mid = int_net(event_tensor, salone_enc_feat_down_detach, salone_enc_feat_mid_detach)   # this is logD, we operate in log space
                    # predict the spectral coefficient logW
                    ddpm_init_res = ddpm_init_net(hsi_blur, salone_enc_feat_down_detach, salone_enc_feat_mid_detach, int_enc_feat_down, int_enc_feat_mid)
                    diffusion.data['int_res'] = int_res
                    diffusion.data['ddpm_init_res'] = ddpm_init_res
                    diffusion.data['condition'] = condition
                    diffusion.data['standalone_enc_feat'] = [salone_enc_feat_down,salone_enc_feat_mid]
                    # diffusion training target
                    diffusion_x0 = torch.log( (hsi_blur)/ (hsi_gt) ) - int_res  - ddpm_init_res
                    diffusion.data['diffusion_x0'] = diffusion_x0
                    diffusion.optimize_parameters_ori()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    while True:
                        try: 
                            diffusion.save_network_deblur(current_epoch, current_step, int_net)
                            break
                        except:
                            print("save_error")
                            time.sleep(10)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
