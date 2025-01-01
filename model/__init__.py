import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

#def create_int_net(opt):
#    from .res_unet import UNetRes as M
#    m = M(in_nc=opt['model']['intergrate_net']['in_channels'],
#          out_nc=opt['model']['intergrate_net']['out_channels'],
#          nc=opt['model']['intergrate_net']['nc'],
#          nb =opt['model']['intergrate_net']['nb']
#         )
#    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
#    return m

def create_int_net(opt):
    model_opt = opt['model']
    from .sr3_modules import unet_light_no_noise_outfeat as unet
    if ('norm_groups' not in model_opt['intergrate_net']) or model_opt['intergrate_net']['norm_groups'] is None:
        model_opt['intergrate_net']['norm_groups']=32
    m = unet.UNet(
        in_channel=model_opt['intergrate_net']['in_channel'],
        out_channel=model_opt['intergrate_net']['out_channel'],
        norm_groups=model_opt['intergrate_net']['norm_groups'],
        inner_channel=model_opt['intergrate_net']['inner_channel'],
        channel_mults=model_opt['intergrate_net']['channel_multiplier'],
        attn_res=model_opt['intergrate_net']['attn_res'],
        res_blocks=model_opt['intergrate_net']['res_blocks'],
        dropout=model_opt['intergrate_net']['dropout'],
        image_size=model_opt['diffusion']['image_size'],
        inner_channel_init=model_opt['standalone_encoder']['inner_channel'],
        channel_mults_init=model_opt['standalone_encoder']['channel_multiplier']
    )
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


def create_ddpm_init_net(opt):
    model_opt = opt['model']
    from .sr3_modules import unet_light_no_noise_twoset_input as unet
    if ('norm_groups' not in model_opt['ddpm_init_net']) or model_opt['ddpm_init_net']['norm_groups'] is None:
        model_opt['ddpm_init_net']['norm_groups']=32
    m = unet.UNet(
        in_channel=model_opt['ddpm_init_net']['in_channel'],
        out_channel=model_opt['ddpm_init_net']['out_channel'],
        norm_groups=model_opt['ddpm_init_net']['norm_groups'],
        inner_channel=model_opt['ddpm_init_net']['inner_channel'],
        channel_mults=model_opt['ddpm_init_net']['channel_multiplier'],
        attn_res=model_opt['ddpm_init_net']['attn_res'],
        res_blocks=model_opt['ddpm_init_net']['res_blocks'],
        dropout=model_opt['ddpm_init_net']['dropout'],
        image_size=model_opt['diffusion']['image_size'],
        noise_net_channel=model_opt['standalone_encoder']['inner_channel'],
        noise_net_channel_mults=model_opt['standalone_encoder']['channel_multiplier'],
        evt_net_channel=model_opt['intergrate_net']['inner_channel'],
        evt_net_channel_mults=model_opt['intergrate_net']['channel_multiplier']
    )
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_standalone_encoder(opt):
    model_opt = opt['model']
    from .sr3_modules import unet_light_no_noise_encoder_only as unet
    if ('norm_groups' not in model_opt['standalone_encoder']) or model_opt['standalone_encoder']['norm_groups'] is None:
        model_opt['standalone_encoder']['norm_groups']=32
    m = unet.UNet(
        in_channel=model_opt['standalone_encoder']['in_channel'],
        out_channel=model_opt['standalone_encoder']['out_channel'],
        norm_groups=model_opt['standalone_encoder']['norm_groups'],
        inner_channel=model_opt['standalone_encoder']['inner_channel'],
        channel_mults=model_opt['standalone_encoder']['channel_multiplier'],
        attn_res=model_opt['standalone_encoder']['attn_res'],
        res_blocks=model_opt['standalone_encoder']['res_blocks'],
        dropout=model_opt['standalone_encoder']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


