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

# crop patch and increase batch size by batch_multiplier times
def random_crop_patch(train_data, patch_size=128, batch_multiplier=16):
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
    parser.add_argument('-c', '--config', type=str, default='config/goprotest.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or test(generation)', default='test')
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
    # integration net
    int_net = diffusion.int_net
    ddpm_init_net = diffusion.ddpm_init_net
    standalone_encoder = diffusion.standalone_encoder
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    patch_crops_per_img = opt['datasets']['train']['patch_crops_per_img']
    batch_multiplier = opt['datasets']['train']['batch_multiplier']
    patch_size = opt['datasets']['train']['patch_size']
    n_iter = n_iter // patch_crops_per_img
    patch_crops_per_img_val = opt['datasets']['val']['patch_crops_per_img']
    patch_size_val = opt['datasets']['val']['patch_size']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['val'])
    if opt['phase'] == 'train':
        pass
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_psnr_blur = 0.0
        avg_ssim_blur = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(test_loader):
            val_data = diffusion.set_device(val_data)
            idx += 1
            gt_img = val_data['hsi_gt']
            val_data['hsi_gt'] = None # a safety
            hsi_blur = val_data['hsi_blur'] # this is B
            event_tensor = val_data['event_bin']

            # pad
            hsi_blur = torch.nn.functional.pad(hsi_blur, (0,0,10,10))
            event_tensor = torch.nn.functional.pad(event_tensor, (0,0,10,10))

            condition = torch.cat([event_tensor,hsi_blur],dim=1)
            diffusion.feed_data(val_data)
            int_net.eval()
            standalone_encoder.eval()
            # get stand alone encoder feat
            with torch.no_grad():
                salone_enc_feat_down,salone_enc_feat_mid = standalone_encoder(condition)
            # detached ver
            salone_enc_feat_down_detach = [f.detach() for f in salone_enc_feat_down]
            salone_enc_feat_mid_detach = [f.detach() for f in salone_enc_feat_mid]
            with torch.no_grad():
                # calculate the integrate result D first
                int_res, int_enc_feat_down, int_enc_feat_mid = int_net(event_tensor, salone_enc_feat_down_detach, salone_enc_feat_mid_detach)   # this is logD
            ddpm_init_net.eval()
            with torch.no_grad():
                ddpm_init_res = ddpm_init_net(hsi_blur, salone_enc_feat_down_detach, salone_enc_feat_mid_detach, int_enc_feat_down, int_enc_feat_mid)
            diffusion.data['int_res'] = int_res
            diffusion.data['ddpm_init_res'] = ddpm_init_res
            diffusion.data['condition'] = condition
            diffusion.data['standalone_enc_feat'] = [salone_enc_feat_down,salone_enc_feat_mid]
            int_res = int_res[:,:,10:-10,:]
            ddpm_init_res = ddpm_init_res[:,:,10:-10,:]
            logW_residual = torch.zeros_like(int_res)
            visuals = diffusion.get_current_visuals_deblur(int_res, ddpm_init_res, logW_residual)
            B_img = Metrics.tensor2img(visuals['B'])  # uint8, HxW
            W_img = Metrics.tensor2img(visuals['W'])  # uint8
            I_img = Metrics.tensor2img(visuals['I'])  # uint8
            L_img = Metrics.tensor2img(visuals['L'])  # uint8
            B_img = B_img[:,:,np.newaxis]
            W_img = W_img[:,:,np.newaxis]
            I_img = I_img[:,:,np.newaxis]

            gt_img = Metrics.tensor2img(gt_img)  # uint8
            gt_img = gt_img[:,:,np.newaxis]
            L_img = L_img[:,:,np.newaxis]

            eval_psnr = Metrics.calculate_psnr(L_img, gt_img)
            eval_ssim = Metrics.calculate_ssim((L_img*255.0).astype(np.uint8), (gt_img*255.0).astype(np.uint8))
            blur_psnr = Metrics.calculate_psnr(B_img, gt_img)
            blur_ssim = Metrics.calculate_ssim((B_img*255.0).astype(np.uint8), (gt_img*255.0).astype(np.uint8))
            logger.info('# Single img Validation # PSNR: {:.4e}'.format(eval_psnr))
            logger.info('# Single img Validation # PSNR blur: {:.4e}'.format(blur_psnr))
            logger.info('# Single img Validation # SSIM: {:.4e}'.format(eval_ssim))
            logger.info('# Single img Validation # SSIM blur: {:.4e}'.format(blur_ssim))

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            avg_psnr_blur += blur_psnr
            avg_ssim_blur += blur_ssim

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_psnr_blur = avg_psnr_blur / idx
        avg_ssim_blur = avg_ssim_blur / idx

        # log
        # Note: we used MATLAB to calculate the PSNR and SSIM in the paper
        # the SSIM value calculated by python here is slightly different than MATLAB's result
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info('# Validation # PSNR blur: {:.4e}'.format(avg_psnr_blur))
        logger.info('# Validation # SSIM blur: {:.4e}'.format(avg_ssim_blur))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim:{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))
