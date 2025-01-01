import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)



class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = False # no SA
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)

    def forward(self, x):
        x = self.res_block(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        inner_channel_init=32,
        channel_mults_init=(1, 2, 4, 8, 8),
    ):
        super().__init__()

        noise_level_channel = None
        self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])
        mid_channel = pre_channel

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        self.align_conv_down = [nn.Conv2d(inner_channel_init, inner_channel,
                           kernel_size=1, padding=0)]

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            channel_mult_init = inner_channel_init * channel_mults_init[ind]
            for _ in range(0, res_blocks):
                self.align_conv_down.append(nn.Conv2d(channel_mult_init, channel_mult,
                               kernel_size=1, padding=0))
                pre_channel = channel_mult
            if not is_last:
                self.align_conv_down.append(nn.Conv2d(channel_mult_init, channel_mult,
                               kernel_size=1, padding=0))
        mid_channel_init = channel_mult_init
            
        self.align_conv_down = nn.ModuleList(self.align_conv_down)


        self.align_conv_mid = [nn.Conv2d(mid_channel_init, mid_channel,
                           kernel_size=1, padding=0),
                               nn.Conv2d(mid_channel_init, mid_channel,
                           kernel_size=1, padding=0),
                              ]
        self.align_conv_mid = nn.ModuleList(self.align_conv_mid)

    def forward(self, x, enc_feats_down, enc_feats_mid):
        feats = []
        out_enc_feats_down = []
        out_enc_feats_mid = []
        idx = 0
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x)
            else:
                x = layer(x)
            enc_f = enc_feats_down[idx]
            align_conv = self.align_conv_down[idx]
            aligned_enc_f = align_conv(enc_f)
            x = x+aligned_enc_f
            out_enc_feats_down.append(x)
            feats.append(x.detach())
            idx = idx + 1

        idx = 0
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x)
            else:
                x = layer(x)
            enc_f = enc_feats_mid[idx]
            align_conv = self.align_conv_mid[idx]
            aligned_enc_f = align_conv(enc_f)
            x = x+aligned_enc_f
            idx = idx + 1
            out_enc_feats_mid.append(x)
        x = x.detach()
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x), out_enc_feats_down, out_enc_feats_mid
