## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
from basicsr.models.archs.wavelet_mamba_blocks import (
    FourDirectionSSMBlock,
    LocalMixBlock,
    WaveletMambaBlock,
    WaveletMambaFusionBiasedBlock,
    WaveletMambaFusionBlock,
    WaveletMambaFusionAnchoredBlock,
    WaveletMambaModulatedBlock,
    WaveletMambaResidualFusionBlock,
    WaveletMambaSpatialFusionBlock,
)


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
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


def build_operator_stage(stage_name,
                         dim,
                         num_heads,
                         num_blocks,
                         ffn_expansion_factor,
                         bias,
                         layer_norm_type,
                         use_wavelet_mamba=False,
                         use_wavelet_mamba_fusion=False,
                         use_wavelet_mamba_fusion_biased=False,
                         use_wavelet_mamba_fusion_anchored=False,
                         use_wavelet_mamba_residual_fusion=False,
                         use_wavelet_mamba_modulation=False,
                         use_wavelet_mamba_spatial_fusion=False,
                         use_ssm_only=False,
                         use_local_mix=False,
                         wavelet_mamba_stages=None,
                         wavelet_mamba_fusion_stages=None,
                         wavelet_mamba_fusion_biased_stages=None,
                         wavelet_mamba_fusion_anchored_stages=None,
                         wavelet_mamba_residual_fusion_stages=None,
                         wavelet_mamba_modulation_stages=None,
                         wavelet_mamba_spatial_fusion_stages=None,
                         ssm_only_stages=None,
                         local_mix_stages=None,
                         local_kernel_sizes=(3, 5)):
    if wavelet_mamba_stages is None:
        wavelet_mamba_stages = []
    if wavelet_mamba_fusion_stages is None:
        wavelet_mamba_fusion_stages = wavelet_mamba_stages if use_wavelet_mamba_fusion else []
    if wavelet_mamba_fusion_biased_stages is None:
        wavelet_mamba_fusion_biased_stages = (
            wavelet_mamba_stages if use_wavelet_mamba_fusion_biased else [])
    if wavelet_mamba_fusion_anchored_stages is None:
        wavelet_mamba_fusion_anchored_stages = (
            wavelet_mamba_stages if use_wavelet_mamba_fusion_anchored else [])
    if wavelet_mamba_residual_fusion_stages is None:
        wavelet_mamba_residual_fusion_stages = (
            wavelet_mamba_stages if use_wavelet_mamba_residual_fusion else [])
    if wavelet_mamba_modulation_stages is None:
        wavelet_mamba_modulation_stages = wavelet_mamba_stages if use_wavelet_mamba_modulation else []
    if wavelet_mamba_spatial_fusion_stages is None:
        wavelet_mamba_spatial_fusion_stages = (
            wavelet_mamba_stages if use_wavelet_mamba_spatial_fusion else [])
    if ssm_only_stages is None:
        ssm_only_stages = wavelet_mamba_stages if use_ssm_only else []
    if local_mix_stages is None:
        local_mix_stages = wavelet_mamba_stages if use_local_mix else []

    use_mamba = use_wavelet_mamba and stage_name in set(wavelet_mamba_stages)
    use_mamba_fusion = use_wavelet_mamba_fusion and stage_name in set(wavelet_mamba_fusion_stages)
    use_mamba_fusion_biased = (
        use_wavelet_mamba_fusion_biased
        and stage_name in set(wavelet_mamba_fusion_biased_stages))
    use_mamba_fusion_anchored = (
        use_wavelet_mamba_fusion_anchored
        and stage_name in set(wavelet_mamba_fusion_anchored_stages))
    use_mamba_residual_fusion = (
        use_wavelet_mamba_residual_fusion
        and stage_name in set(wavelet_mamba_residual_fusion_stages))
    use_mamba_modulation = (
        use_wavelet_mamba_modulation and stage_name in set(wavelet_mamba_modulation_stages))
    use_mamba_spatial_fusion = (
        use_wavelet_mamba_spatial_fusion
        and stage_name in set(wavelet_mamba_spatial_fusion_stages))
    use_ssm = use_ssm_only and stage_name in set(ssm_only_stages)
    use_local = use_local_mix and stage_name in set(local_mix_stages)
    if use_mamba_spatial_fusion:
        return nn.Sequential(*[
            WaveletMambaSpatialFusionBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=layer_norm_type,
                local_kernel_sizes=local_kernel_sizes) for _ in range(num_blocks)
        ])
    if use_mamba_modulation:
        return nn.Sequential(*[
            WaveletMambaModulatedBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=layer_norm_type,
                local_kernel_sizes=local_kernel_sizes) for _ in range(num_blocks)
        ])
    if use_mamba_residual_fusion:
        return nn.Sequential(*[
            WaveletMambaResidualFusionBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=layer_norm_type,
                local_kernel_sizes=local_kernel_sizes) for _ in range(num_blocks)
        ])
    if use_mamba_fusion_biased:
        return nn.Sequential(*[
            WaveletMambaFusionBiasedBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=layer_norm_type,
                local_kernel_sizes=local_kernel_sizes) for _ in range(num_blocks)
        ])
    if use_mamba_fusion_anchored:
        return nn.Sequential(*[
            WaveletMambaFusionAnchoredBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=layer_norm_type,
                local_kernel_sizes=local_kernel_sizes) for _ in range(num_blocks)
        ])
    if use_mamba_fusion:
        return nn.Sequential(*[
            WaveletMambaFusionBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=layer_norm_type,
                local_kernel_sizes=local_kernel_sizes) for _ in range(num_blocks)
        ])
    if use_mamba:
        return nn.Sequential(*[
            WaveletMambaBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=layer_norm_type,
                local_kernel_sizes=local_kernel_sizes) for _ in range(num_blocks)
        ])
    if use_ssm:
        return nn.Sequential(*[
            FourDirectionSSMBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=layer_norm_type) for _ in range(num_blocks)
        ])
    if use_local:
        return nn.Sequential(*[
            LocalMixBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=layer_norm_type,
                local_kernel_sizes=local_kernel_sizes) for _ in range(num_blocks)
        ])

    return nn.Sequential(*[
        TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=layer_norm_type) for _ in range(num_blocks)
    ])


class Calibra(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=3, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Calibra, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        

        x_1 = self.conv1(img)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        out = img+img*illu_map
        return out




##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class WTNet(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 40,
        num_blocks = [4,7,7,8], 
        num_refinement_blocks = 2,
        heads = [1,1,1,1],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        use_wavelet_mamba = False,
        use_wavelet_mamba_fusion = False,
        use_wavelet_mamba_fusion_biased = False,
        use_wavelet_mamba_fusion_anchored = False,
        use_wavelet_mamba_residual_fusion = False,
        use_wavelet_mamba_modulation = False,
        use_wavelet_mamba_spatial_fusion = False,
        use_ssm_only = False,
        use_local_mix = False,
        wavelet_mamba_stages = None,
        wavelet_mamba_fusion_stages = None,
        wavelet_mamba_fusion_biased_stages = None,
        wavelet_mamba_fusion_anchored_stages = None,
        wavelet_mamba_residual_fusion_stages = None,
        wavelet_mamba_modulation_stages = None,
        wavelet_mamba_spatial_fusion_stages = None,
        ssm_only_stages = None,
        local_mix_stages = None,
        local_kernel_sizes = (3, 5)
    ):

        super(WTNet, self).__init__()

        self.cali = Calibra(dim)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.xfm = DWTForward(J=1, mode='zero',wave='haar')
        self.ifm = DWTInverse(mode='zero', wave='haar')
        self.dwc1 = nn.Conv2d(dim*3,dim*3,3,1,1,groups=dim*3,bias=False)
        self.dwc2 = nn.Conv2d(dim*3,dim*3,3,1,1,groups=dim*3,bias=False)
        self.dwc3 = nn.Conv2d(dim*3,dim*3,3,1,1,groups=dim*3,bias=False)
        self.encoder_level1 = build_operator_stage(
            stage_name='encoder_level1',
            dim=dim,
            num_heads=heads[0],
            num_blocks=num_blocks[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=LayerNorm_type,
            use_wavelet_mamba=use_wavelet_mamba,
            use_wavelet_mamba_fusion=use_wavelet_mamba_fusion,
            use_wavelet_mamba_fusion_biased=use_wavelet_mamba_fusion_biased,
            use_wavelet_mamba_fusion_anchored=use_wavelet_mamba_fusion_anchored,
            use_wavelet_mamba_residual_fusion=use_wavelet_mamba_residual_fusion,
            use_wavelet_mamba_modulation=use_wavelet_mamba_modulation,
            use_wavelet_mamba_spatial_fusion=use_wavelet_mamba_spatial_fusion,
            use_ssm_only=use_ssm_only,
            use_local_mix=use_local_mix,
            wavelet_mamba_stages=wavelet_mamba_stages,
            wavelet_mamba_fusion_stages=wavelet_mamba_fusion_stages,
            wavelet_mamba_fusion_biased_stages=wavelet_mamba_fusion_biased_stages,
            wavelet_mamba_fusion_anchored_stages=wavelet_mamba_fusion_anchored_stages,
            wavelet_mamba_residual_fusion_stages=wavelet_mamba_residual_fusion_stages,
            wavelet_mamba_modulation_stages=wavelet_mamba_modulation_stages,
            wavelet_mamba_spatial_fusion_stages=wavelet_mamba_spatial_fusion_stages,
            ssm_only_stages=ssm_only_stages,
            local_mix_stages=local_mix_stages,
            local_kernel_sizes=local_kernel_sizes)
        
        self.encoder_level2 = build_operator_stage(
            stage_name='encoder_level2',
            dim=dim,
            num_heads=heads[1],
            num_blocks=num_blocks[1],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=LayerNorm_type,
            use_wavelet_mamba=use_wavelet_mamba,
            use_wavelet_mamba_fusion=use_wavelet_mamba_fusion,
            use_wavelet_mamba_fusion_biased=use_wavelet_mamba_fusion_biased,
            use_wavelet_mamba_fusion_anchored=use_wavelet_mamba_fusion_anchored,
            use_wavelet_mamba_residual_fusion=use_wavelet_mamba_residual_fusion,
            use_wavelet_mamba_modulation=use_wavelet_mamba_modulation,
            use_wavelet_mamba_spatial_fusion=use_wavelet_mamba_spatial_fusion,
            use_ssm_only=use_ssm_only,
            use_local_mix=use_local_mix,
            wavelet_mamba_stages=wavelet_mamba_stages,
            wavelet_mamba_fusion_stages=wavelet_mamba_fusion_stages,
            wavelet_mamba_fusion_biased_stages=wavelet_mamba_fusion_biased_stages,
            wavelet_mamba_fusion_anchored_stages=wavelet_mamba_fusion_anchored_stages,
            wavelet_mamba_residual_fusion_stages=wavelet_mamba_residual_fusion_stages,
            wavelet_mamba_modulation_stages=wavelet_mamba_modulation_stages,
            wavelet_mamba_spatial_fusion_stages=wavelet_mamba_spatial_fusion_stages,
            ssm_only_stages=ssm_only_stages,
            local_mix_stages=local_mix_stages,
            local_kernel_sizes=local_kernel_sizes)
        self.encoder_level3 = build_operator_stage(
            stage_name='encoder_level3',
            dim=dim,
            num_heads=heads[1],
            num_blocks=num_blocks[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=LayerNorm_type,
            use_wavelet_mamba=use_wavelet_mamba,
            use_wavelet_mamba_fusion=use_wavelet_mamba_fusion,
            use_wavelet_mamba_fusion_biased=use_wavelet_mamba_fusion_biased,
            use_wavelet_mamba_fusion_anchored=use_wavelet_mamba_fusion_anchored,
            use_wavelet_mamba_residual_fusion=use_wavelet_mamba_residual_fusion,
            use_wavelet_mamba_modulation=use_wavelet_mamba_modulation,
            use_wavelet_mamba_spatial_fusion=use_wavelet_mamba_spatial_fusion,
            use_ssm_only=use_ssm_only,
            use_local_mix=use_local_mix,
            wavelet_mamba_stages=wavelet_mamba_stages,
            wavelet_mamba_fusion_stages=wavelet_mamba_fusion_stages,
            wavelet_mamba_fusion_biased_stages=wavelet_mamba_fusion_biased_stages,
            wavelet_mamba_fusion_anchored_stages=wavelet_mamba_fusion_anchored_stages,
            wavelet_mamba_residual_fusion_stages=wavelet_mamba_residual_fusion_stages,
            wavelet_mamba_modulation_stages=wavelet_mamba_modulation_stages,
            wavelet_mamba_spatial_fusion_stages=wavelet_mamba_spatial_fusion_stages,
            ssm_only_stages=ssm_only_stages,
            local_mix_stages=local_mix_stages,
            local_kernel_sizes=local_kernel_sizes)
      
        self.latent = build_operator_stage(
            stage_name='latent',
            dim=dim,
            num_heads=heads[3],
            num_blocks=num_blocks[3],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=LayerNorm_type,
            use_wavelet_mamba=use_wavelet_mamba,
            use_wavelet_mamba_fusion=use_wavelet_mamba_fusion,
            use_wavelet_mamba_fusion_biased=use_wavelet_mamba_fusion_biased,
            use_wavelet_mamba_fusion_anchored=use_wavelet_mamba_fusion_anchored,
            use_wavelet_mamba_residual_fusion=use_wavelet_mamba_residual_fusion,
            use_wavelet_mamba_modulation=use_wavelet_mamba_modulation,
            use_wavelet_mamba_spatial_fusion=use_wavelet_mamba_spatial_fusion,
            use_ssm_only=use_ssm_only,
            use_local_mix=use_local_mix,
            wavelet_mamba_stages=wavelet_mamba_stages,
            wavelet_mamba_fusion_stages=wavelet_mamba_fusion_stages,
            wavelet_mamba_fusion_biased_stages=wavelet_mamba_fusion_biased_stages,
            wavelet_mamba_fusion_anchored_stages=wavelet_mamba_fusion_anchored_stages,
            wavelet_mamba_residual_fusion_stages=wavelet_mamba_residual_fusion_stages,
            wavelet_mamba_modulation_stages=wavelet_mamba_modulation_stages,
            wavelet_mamba_spatial_fusion_stages=wavelet_mamba_spatial_fusion_stages,
            ssm_only_stages=ssm_only_stages,
            local_mix_stages=local_mix_stages,
            local_kernel_sizes=local_kernel_sizes)
        
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level3 = build_operator_stage(
            stage_name='decoder_level3',
            dim=int(dim),
            num_heads=heads[2],
            num_blocks=num_blocks[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=LayerNorm_type,
            use_wavelet_mamba=use_wavelet_mamba,
            use_wavelet_mamba_fusion=use_wavelet_mamba_fusion,
            use_wavelet_mamba_fusion_biased=use_wavelet_mamba_fusion_biased,
            use_wavelet_mamba_fusion_anchored=use_wavelet_mamba_fusion_anchored,
            use_wavelet_mamba_residual_fusion=use_wavelet_mamba_residual_fusion,
            use_wavelet_mamba_modulation=use_wavelet_mamba_modulation,
            use_wavelet_mamba_spatial_fusion=use_wavelet_mamba_spatial_fusion,
            use_ssm_only=use_ssm_only,
            use_local_mix=use_local_mix,
            wavelet_mamba_stages=wavelet_mamba_stages,
            wavelet_mamba_fusion_stages=wavelet_mamba_fusion_stages,
            wavelet_mamba_fusion_biased_stages=wavelet_mamba_fusion_biased_stages,
            wavelet_mamba_fusion_anchored_stages=wavelet_mamba_fusion_anchored_stages,
            wavelet_mamba_residual_fusion_stages=wavelet_mamba_residual_fusion_stages,
            wavelet_mamba_modulation_stages=wavelet_mamba_modulation_stages,
            wavelet_mamba_spatial_fusion_stages=wavelet_mamba_spatial_fusion_stages,
            ssm_only_stages=ssm_only_stages,
            local_mix_stages=local_mix_stages,
            local_kernel_sizes=local_kernel_sizes)

        self.reduce_chan_level2 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level2 = build_operator_stage(
            stage_name='decoder_level2',
            dim=int(dim),
            num_heads=heads[1],
            num_blocks=num_blocks[1],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=LayerNorm_type,
            use_wavelet_mamba=use_wavelet_mamba,
            use_wavelet_mamba_fusion=use_wavelet_mamba_fusion,
            use_wavelet_mamba_fusion_biased=use_wavelet_mamba_fusion_biased,
            use_wavelet_mamba_fusion_anchored=use_wavelet_mamba_fusion_anchored,
            use_wavelet_mamba_residual_fusion=use_wavelet_mamba_residual_fusion,
            use_wavelet_mamba_modulation=use_wavelet_mamba_modulation,
            use_wavelet_mamba_spatial_fusion=use_wavelet_mamba_spatial_fusion,
            use_ssm_only=use_ssm_only,
            use_local_mix=use_local_mix,
            wavelet_mamba_stages=wavelet_mamba_stages,
            wavelet_mamba_fusion_stages=wavelet_mamba_fusion_stages,
            wavelet_mamba_fusion_biased_stages=wavelet_mamba_fusion_biased_stages,
            wavelet_mamba_fusion_anchored_stages=wavelet_mamba_fusion_anchored_stages,
            wavelet_mamba_residual_fusion_stages=wavelet_mamba_residual_fusion_stages,
            wavelet_mamba_modulation_stages=wavelet_mamba_modulation_stages,
            wavelet_mamba_spatial_fusion_stages=wavelet_mamba_spatial_fusion_stages,
            ssm_only_stages=ssm_only_stages,
            local_mix_stages=local_mix_stages,
            local_kernel_sizes=local_kernel_sizes)
        
        self.decoder_level1 = build_operator_stage(
            stage_name='decoder_level1',
            dim=int(dim*2**1),
            num_heads=heads[0],
            num_blocks=num_blocks[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=LayerNorm_type,
            use_wavelet_mamba=use_wavelet_mamba,
            use_wavelet_mamba_fusion=use_wavelet_mamba_fusion,
            use_wavelet_mamba_fusion_anchored=use_wavelet_mamba_fusion_anchored,
            use_wavelet_mamba_residual_fusion=use_wavelet_mamba_residual_fusion,
            use_wavelet_mamba_modulation=use_wavelet_mamba_modulation,
            use_wavelet_mamba_spatial_fusion=use_wavelet_mamba_spatial_fusion,
            use_ssm_only=use_ssm_only,
            use_local_mix=use_local_mix,
            wavelet_mamba_stages=wavelet_mamba_stages,
            wavelet_mamba_fusion_stages=wavelet_mamba_fusion_stages,
            wavelet_mamba_fusion_anchored_stages=wavelet_mamba_fusion_anchored_stages,
            wavelet_mamba_residual_fusion_stages=wavelet_mamba_residual_fusion_stages,
            wavelet_mamba_modulation_stages=wavelet_mamba_modulation_stages,
            wavelet_mamba_spatial_fusion_stages=wavelet_mamba_spatial_fusion_stages,
            ssm_only_stages=ssm_only_stages,
            local_mix_stages=local_mix_stages,
            local_kernel_sizes=local_kernel_sizes)
        
        self.refinement = build_operator_stage(
            stage_name='refinement',
            dim=int(dim*2**1),
            num_heads=heads[0],
            num_blocks=num_refinement_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=LayerNorm_type,
            use_wavelet_mamba=use_wavelet_mamba,
            use_wavelet_mamba_fusion=use_wavelet_mamba_fusion,
            use_wavelet_mamba_residual_fusion=use_wavelet_mamba_residual_fusion,
            use_wavelet_mamba_modulation=use_wavelet_mamba_modulation,
            use_wavelet_mamba_spatial_fusion=use_wavelet_mamba_spatial_fusion,
            use_ssm_only=use_ssm_only,
            use_local_mix=use_local_mix,
            wavelet_mamba_stages=wavelet_mamba_stages,
            wavelet_mamba_fusion_stages=wavelet_mamba_fusion_stages,
            wavelet_mamba_residual_fusion_stages=wavelet_mamba_residual_fusion_stages,
            wavelet_mamba_modulation_stages=wavelet_mamba_modulation_stages,
            wavelet_mamba_spatial_fusion_stages=wavelet_mamba_spatial_fusion_stages,
            ssm_only_stages=ssm_only_stages,
            local_mix_stages=local_mix_stages,
            local_kernel_sizes=local_kernel_sizes)
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)



    def forward(self, inp_img):

        inp_img = self.cali(inp_img)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        l_inp_enc_level2, h_inp_enc_level2=self.xfm(out_enc_level1)
        
        h2_shape = h_inp_enc_level2[0].shape
        h2 = h_inp_enc_level2[0].reshape(h2_shape[0],h2_shape[1]*3,h2_shape[3],h2_shape[4])
        h2 = self.dwc1(h2)
        h2 = h2.reshape(h2_shape)
        h_inp_enc_level2 = [h2]
        out_enc_level2 = self.encoder_level2(l_inp_enc_level2)

        l_inp_enc_level3, h_inp_enc_level3=self.xfm(out_enc_level2)

        h3_shape = h_inp_enc_level3[0].shape
        h3 = h_inp_enc_level3[0].reshape(h3_shape[0],h3_shape[1]*3,h3_shape[3],h3_shape[4])
        h3 = self.dwc2(h3)
        h3 = h3.reshape(h3_shape)
        h_inp_enc_level3 = [h3]

        out_enc_level3 = self.encoder_level3(l_inp_enc_level3)

        l_inp_enc_level4, h_inp_enc_level4=self.xfm(out_enc_level3)

        h4_shape = h_inp_enc_level4[0].shape
        h4 = h_inp_enc_level4[0].reshape(h4_shape[0],h4_shape[1]*3,h4_shape[3],h4_shape[4])
        h4 = self.dwc3(h4)
        h4 = h4.reshape(h4_shape)
        h_inp_enc_level4 = [h4]
               
        latent = self.latent(l_inp_enc_level4) 

        inp_dec_level3 = self.ifm((latent,h_inp_enc_level4))
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        
        inp_dec_level2 = self.ifm((out_dec_level3,h_inp_enc_level3))
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.ifm((out_dec_level2, h_inp_enc_level2))
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1


class WTNetMamba(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 wavelet_mamba_stages=('encoder_level2', 'encoder_level3', 'latent'),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=True,
            use_wavelet_mamba_fusion=False,
            use_wavelet_mamba_modulation=False,
            use_wavelet_mamba_spatial_fusion=False,
            use_local_mix=False,
            wavelet_mamba_stages=wavelet_mamba_stages,
            local_kernel_sizes=local_kernel_sizes)


class WTNetLocalMix(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 wavelet_mamba_stages=('encoder_level2', 'encoder_level3', 'latent'),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=False,
            use_wavelet_mamba_fusion=False,
            use_wavelet_mamba_modulation=False,
            use_wavelet_mamba_spatial_fusion=False,
            use_ssm_only=False,
            use_local_mix=True,
            wavelet_mamba_stages=wavelet_mamba_stages,
            local_kernel_sizes=local_kernel_sizes)


class WTNetSSM(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 wavelet_mamba_stages=('encoder_level2', 'encoder_level3', 'latent'),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=False,
            use_wavelet_mamba_fusion=False,
            use_wavelet_mamba_modulation=False,
            use_wavelet_mamba_spatial_fusion=False,
            use_ssm_only=True,
            use_local_mix=False,
            wavelet_mamba_stages=wavelet_mamba_stages,
            local_kernel_sizes=local_kernel_sizes)


class WTNetMambaFusion(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 wavelet_mamba_stages=('encoder_level2', 'encoder_level3', 'latent'),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=False,
            use_wavelet_mamba_fusion=True,
            use_wavelet_mamba_fusion_biased=False,
            use_wavelet_mamba_fusion_anchored=False,
            use_wavelet_mamba_residual_fusion=False,
            use_wavelet_mamba_modulation=False,
            use_wavelet_mamba_spatial_fusion=False,
            use_ssm_only=False,
            use_local_mix=False,
            wavelet_mamba_stages=wavelet_mamba_stages,
            local_kernel_sizes=local_kernel_sizes)


class WTNetMambaModulated(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 wavelet_mamba_stages=('encoder_level2', 'encoder_level3', 'latent'),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=False,
            use_wavelet_mamba_fusion=False,
            use_wavelet_mamba_fusion_biased=False,
            use_wavelet_mamba_fusion_anchored=False,
            use_wavelet_mamba_residual_fusion=False,
            use_wavelet_mamba_modulation=True,
            use_wavelet_mamba_spatial_fusion=False,
            use_ssm_only=False,
            use_local_mix=False,
            wavelet_mamba_stages=wavelet_mamba_stages,
            local_kernel_sizes=local_kernel_sizes)


class WTNetMambaSpatialFusion(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 wavelet_mamba_stages=('encoder_level2', 'encoder_level3', 'latent'),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=False,
            use_wavelet_mamba_fusion=False,
            use_wavelet_mamba_fusion_biased=False,
            use_wavelet_mamba_fusion_anchored=False,
            use_wavelet_mamba_residual_fusion=False,
            use_wavelet_mamba_modulation=False,
            use_wavelet_mamba_spatial_fusion=True,
            use_ssm_only=False,
            use_local_mix=False,
            wavelet_mamba_stages=wavelet_mamba_stages,
            local_kernel_sizes=local_kernel_sizes)


class WTNetMambaSelectiveFusion(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 fusion_stages=('encoder_level3', 'latent'),
                 ssm_stages=('encoder_level2',),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=False,
            use_wavelet_mamba_fusion=True,
            use_wavelet_mamba_fusion_biased=False,
            use_wavelet_mamba_fusion_anchored=False,
            use_wavelet_mamba_residual_fusion=False,
            use_wavelet_mamba_modulation=False,
            use_wavelet_mamba_spatial_fusion=False,
            use_ssm_only=True,
            use_local_mix=False,
            wavelet_mamba_fusion_stages=fusion_stages,
            ssm_only_stages=ssm_stages,
            local_kernel_sizes=local_kernel_sizes)


class WTNetMambaResidualFusion(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 wavelet_mamba_stages=('encoder_level2', 'encoder_level3', 'latent'),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=False,
            use_wavelet_mamba_fusion=False,
            use_wavelet_mamba_fusion_biased=False,
            use_wavelet_mamba_fusion_anchored=False,
            use_wavelet_mamba_residual_fusion=True,
            use_wavelet_mamba_modulation=False,
            use_wavelet_mamba_spatial_fusion=False,
            use_ssm_only=False,
            use_local_mix=False,
            wavelet_mamba_stages=wavelet_mamba_stages,
            local_kernel_sizes=local_kernel_sizes)


class WTNetMambaFusionAnchored(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 wavelet_mamba_stages=('encoder_level2', 'encoder_level3', 'latent'),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=False,
            use_wavelet_mamba_fusion=False,
            use_wavelet_mamba_fusion_biased=False,
            use_wavelet_mamba_fusion_anchored=True,
            use_wavelet_mamba_residual_fusion=False,
            use_wavelet_mamba_modulation=False,
            use_wavelet_mamba_spatial_fusion=False,
            use_ssm_only=False,
            use_local_mix=False,
            wavelet_mamba_stages=wavelet_mamba_stages,
            local_kernel_sizes=local_kernel_sizes)


class WTNetMambaFusionBiased(WTNet):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[4, 7, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 wavelet_mamba_stages=('encoder_level2', 'encoder_level3', 'latent'),
                 local_kernel_sizes=(3, 5)):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            use_wavelet_mamba=False,
            use_wavelet_mamba_fusion=False,
            use_wavelet_mamba_fusion_biased=True,
            use_wavelet_mamba_fusion_anchored=False,
            use_wavelet_mamba_residual_fusion=False,
            use_wavelet_mamba_modulation=False,
            use_wavelet_mamba_spatial_fusion=False,
            use_ssm_only=False,
            use_local_mix=False,
            wavelet_mamba_stages=wavelet_mamba_stages,
            local_kernel_sizes=local_kernel_sizes)

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    model = WTNet().cuda()
    print(model)
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')
