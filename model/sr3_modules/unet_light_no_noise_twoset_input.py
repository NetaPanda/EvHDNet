import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange
import numbers


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
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


#########################################

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1, bias=bias)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32, sf=None, num_heads=1):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.sf = sf
        if self.sf is not None:
            self.s2att = MultiheadSSCA(dim_out,sf,num_heads=num_heads)

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.block2(h)
        if self.sf is not None:
            h = self.s2att(h)
        return h + self.res_conv(x)

class SpatialSpectralCrossAttention(nn.Module):
    def __init__(self, dim, downscale_ratio=4, squeeze_ratio=16):
        super().__init__()
        # common components
        self.sigmoid = nn.Sigmoid()
        self.unshuffle =nn.PixelUnshuffle(downscale_ratio)
        self.shuffle =nn.PixelShuffle(downscale_ratio)
        # channel attention
        cdim = dim * downscale_ratio * downscale_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(cdim, cdim // squeeze_ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(cdim // squeeze_ratio, cdim, 1, bias=False))
        self.tospatial = nn.AvgPool1d(downscale_ratio * downscale_ratio, stride=downscale_ratio * downscale_ratio)

        # spatial attention
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.tospectral = nn.AvgPool2d(downscale_ratio, downscale_ratio)
        self.spa_temperature = nn.Parameter(torch.ones(1,))
        self.spe_temperature = nn.Parameter(torch.ones(1,))

       
    def forward(self, x):
        b, c, h, w = x.shape
        # channel attention from spectral features
        x_spectral = self.unshuffle(x)
        avg_out = self.fc(self.avg_pool(x_spectral))
        max_out = self.fc(self.max_pool(x_spectral))
        out = avg_out + max_out
        catt_map = self.sigmoid(out) * self.spe_temperature
        # process both spatial and spectral features with channel attention map
        x_spectral = catt_map * x_spectral
        scatt_map = self.tospatial(catt_map.squeeze(-1).squeeze(-1))
        scatt_map = scatt_map.unsqueeze(-1).unsqueeze(-1)
        x = scatt_map * x
        
        # spatial attention from spatial features
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        satt_map = torch.cat([avg_out, max_out], dim=1)
        satt_map = self.conv1(satt_map)
        satt_map = self.sigmoid(satt_map) * self.spa_temperature
        # process both spatial and spectral features with spatial attention map
        x = satt_map * x
        csatt_map = self.tospectral(satt_map)
        x_spectral = csatt_map * x_spectral
        
        # finally add spatial and spectral features
        x_spectral = self.shuffle(x_spectral)
        x = (x + x_spectral) / 2.0
        return x 


class MultiheadSSCA(nn.Module):
    def __init__(self, dim, downscale_ratio=4, squeeze_ratio=2, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        dim_ssca = dim // num_heads
        self.SSCAs = [SpatialSpectralCrossAttention(dim_ssca, downscale_ratio, squeeze_ratio) for i in range(num_heads)]
        self.SSCAs = nn.ModuleList(self.SSCAs)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x_split = rearrange(x, 'b (head c) h w -> b head c h w', head=self.num_heads)
        ssca_out_list = []
        for i,ssca_module in enumerate(self.SSCAs):
            ssca_input = x_split[:,i,:,:,:]
            ssca_output = ssca_module(ssca_input)
            ssca_out_list.append(ssca_output)
        x_out = torch.cat(ssca_out_list, dim=1)
        x_out = self.project_out(x_out)
        return x_out



class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False, sf=None, num_heads=1):
        super().__init__()
        self.with_attn = False # no SA
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout,sf=sf,num_heads=num_heads)

    def forward(self, x):
        x = self.res_block(x)
        return x

class EventImgFusionBlock(nn.Module):
    def __init__(self, dim_evt, dim_img, ffn_expansion_factor=2):
        super().__init__()
        self.evt_prefuse_1x1 = nn.Conv2d(dim_evt, dim_img , 1, 1, 0)
        self.img_prefuse_1x1 = nn.Conv2d(dim_img, dim_img , 1, 1, 0)
        self.LN_img = LayerNorm(dim_img)
        self.LN_evt = LayerNorm(dim_img)
        mlp_hidden_dim = int(dim_img * ffn_expansion_factor)
        self.mlp_gating = Mlp(in_features=dim_img, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.gating_sigmoid = nn.Sigmoid()
        self.spatial_gating_conv = nn.Conv2d(dim_img, dim_img, 1, 1, 0)

        self.mlp_fuse = Mlp(in_features=dim_img, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.LN_MLP = nn.LayerNorm(dim_img)

    def forward(self, x_img, x_evt):
        b, c, h, w = x_img.shape
        x_img = self.img_prefuse_1x1(x_img)
        x_evt = self.evt_prefuse_1x1(x_evt)

        x_img_norm = self.LN_img(x_img) 
        x_evt_norm = self.LN_evt(x_evt) 
        x_img_norm = to_3d(x_img_norm) # b, h*w, c
        x_img_mutual_gate = x_img_norm * self.gating_sigmoid(self.mlp_gating(x_img_norm))
        x_img_mutual_gate = to_4d(x_img_mutual_gate, h, w) # b, c, h, w
        x_evt_norm = to_3d(x_evt_norm) # b, h*w, c
        x_evt_mutual_gate = x_evt_norm * self.gating_sigmoid(self.mlp_gating(x_evt_norm))
        x_evt_mutual_gate = to_4d(x_evt_mutual_gate, h, w) # b, c, h, w

        spatial_gate = self.gating_sigmoid(self.spatial_gating_conv(x_img_mutual_gate))
        x_evt_mutual_gate = x_evt_mutual_gate * spatial_gate

        fused = x_img + x_evt_mutual_gate 
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.mlp_fuse(self.LN_MLP(fused))
        fused = to_4d(fused, h, w)
        
        return fused


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
        noise_net_channel=32,
        noise_net_channel_mults=(1, 2, 4, 8, 8),
        evt_net_channel=32,
        evt_net_channel_mults=(1, 2, 4, 8, 8)
    ):
        super().__init__()

        noise_level_channel = None
        self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        s2_scale_factor = 4 
        num_heads = 1
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        # note that all downs and mids layers have 2x input channels, if we use concat fusion
        # if use align_conv_1, then change back to 1x (delete the "2*")
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn, sf=s2_scale_factor,num_heads=num_heads))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel,pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
                s2_scale_factor = s2_scale_factor // 2
                if s2_scale_factor < 1:
                    s2_scale_factor = 1
                num_heads = num_heads * 2

        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True, sf=s2_scale_factor, num_heads=num_heads),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False, sf=s2_scale_factor, num_heads=num_heads)
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

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups, bias=True)


        # noise fusion modules
        # noise fusion is just a 1x1 conv and add operation
        self.fuse_or_not_down = [False] # dont fuse the first conv
        self.align_conv_down = []

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            channel_mult_init = noise_net_channel * noise_net_channel_mults[ind]
            for _ in range(0, res_blocks):
                self.align_conv_down.append(nn.Conv2d(channel_mult_init, channel_mult,
                               kernel_size=1, padding=0))
                self.fuse_or_not_down.append(True)
                pre_channel = channel_mult
            if not is_last:
                self.fuse_or_not_down.append(False)

        mid_channel_init = channel_mult_init
            
        self.align_conv_down = nn.ModuleList(self.align_conv_down)


        self.align_conv_mid = [nn.Conv2d(mid_channel_init, mid_channel,
                           kernel_size=1, padding=0),
                               nn.Conv2d(mid_channel_init, mid_channel,
                           kernel_size=1, padding=0),
                              ]
        self.align_conv_mid = nn.ModuleList(self.align_conv_mid)
        # noise fusion end

      
        # event fusion modules
        self.fuse_or_not_down_1 = [False]
        self.align_conv_down_1 = []

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            channel_mult_init = evt_net_channel * evt_net_channel_mults[ind]
            for _ in range(0, res_blocks):
                self.align_conv_down_1.append(EventImgFusionBlock(channel_mult_init, channel_mult))
                self.fuse_or_not_down_1.append(True)
                pre_channel = channel_mult
            if not is_last:
                self.fuse_or_not_down_1.append(False)
        mid_channel_init = channel_mult_init
            
        self.align_conv_down_1 = nn.ModuleList(self.align_conv_down_1)


        self.align_conv_mid_1 = [EventImgFusionBlock(mid_channel_init, mid_channel),
                                 EventImgFusionBlock(mid_channel_init, mid_channel)
                                ]
        self.align_conv_mid_1 = nn.ModuleList(self.align_conv_mid_1)


    # enc_feats=noise feats enc_feats_1=evt feats
    def forward(self, x, enc_feats_down, enc_feats_mid, enc_feats_down_1, enc_feats_mid_1):
        feats = []
        idx = 0
        align_idx = 0
        align_idx_1 = 0
        for layer in self.downs:
            x = layer(x)
            # fuse evt
            if self.fuse_or_not_down_1[idx]:
                enc_f_1 = enc_feats_down_1[idx]
                align_conv_1 = self.align_conv_down_1[align_idx_1]
                x = align_conv_1(x, enc_f_1)
                align_idx_1 = align_idx_1 + 1

            # fuse noise
            if self.fuse_or_not_down[idx]:
                enc_f = enc_feats_down[idx]
                align_conv = self.align_conv_down[align_idx]
                aligned_enc_f = align_conv(enc_f)
                x = x+aligned_enc_f
                align_idx = align_idx + 1

            feats.append(x)
            idx = idx + 1

        idx = 0
        for layer in self.mid:
            x = layer(x)

            # fuse evt
            enc_f_1 = enc_feats_mid_1[idx]
            align_conv_1 = self.align_conv_mid_1[idx]
            x = align_conv_1(x, enc_f_1)

            # fuse noise
            enc_f = enc_feats_mid[idx]
            align_conv = self.align_conv_mid[idx]
            aligned_enc_f = align_conv(enc_f)
            x = x+aligned_enc_f
            idx = idx + 1

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x)
