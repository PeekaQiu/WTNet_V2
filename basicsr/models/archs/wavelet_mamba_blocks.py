import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class LocalMixBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 local_kernel_sizes=(3, 5)):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.local_mix = LocalMixing(dim, list(local_kernel_sizes), bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.local_mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FourDirectionSSMBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.four_direction_ssm = FourDirectionSSMMixing(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.four_direction_ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LocalMixing(nn.Module):
    def __init__(self, dim, kernel_sizes, bias):
        super().__init__()
        if not kernel_sizes:
            raise ValueError('kernel_sizes must contain at least one kernel size.')

        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(
                dim,
                dim,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=dim,
                bias=bias) for k in kernel_sizes
        ])
        self.fuse = nn.Conv2d(dim * len(kernel_sizes), dim * 2, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        mixed = [conv(x) for conv in self.depthwise_convs]
        mixed = self.fuse(torch.cat(mixed, dim=1))
        x1, x2 = mixed.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * torch.sigmoid(x2))


class DirectionalStateSpace(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.decay = nn.Parameter(torch.zeros(dim))

    def forward(self, seq):
        batch_size, seq_len, dim = seq.shape
        decay = torch.sigmoid(self.decay).view(1, dim)
        state = seq.new_zeros(batch_size, dim)
        outputs = []

        for step in range(seq_len):
            value, gate = self.in_proj(seq[:, step, :]).chunk(2, dim=-1)
            value = torch.tanh(value)
            gate = torch.sigmoid(gate)
            state = decay * state + (1.0 - decay) * value
            outputs.append(self.out_proj(gate * state + (1.0 - gate) * seq[:, step, :]))

        return torch.stack(outputs, dim=1)


class FourDirectionSSMMixing(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.left_to_right = DirectionalStateSpace(dim)
        self.right_to_left = DirectionalStateSpace(dim)
        self.top_to_bottom = DirectionalStateSpace(dim)
        self.bottom_to_top = DirectionalStateSpace(dim)
        self.fuse = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)
        self.gate = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)

    def _scan_width(self, x, mixer, reverse=False):
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
        if reverse:
            seq = torch.flip(seq, dims=[1])
        out = mixer(seq)
        if reverse:
            out = torch.flip(out, dims=[1])
        return out.reshape(b, h, w, c).permute(0, 3, 1, 2)

    def _scan_height(self, x, mixer, reverse=False):
        b, c, h, w = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        if reverse:
            seq = torch.flip(seq, dims=[1])
        out = mixer(seq)
        if reverse:
            out = torch.flip(out, dims=[1])
        return out.reshape(b, w, h, c).permute(0, 3, 2, 1)

    def forward(self, x):
        lr = self._scan_width(x, self.left_to_right, reverse=False)
        rl = self._scan_width(x, self.right_to_left, reverse=True)
        tb = self._scan_height(x, self.top_to_bottom, reverse=False)
        bt = self._scan_height(x, self.bottom_to_top, reverse=True)
        mixed = torch.cat([lr, rl, tb, bt], dim=1)
        return self.fuse(mixed) * torch.sigmoid(self.gate(mixed))


class WaveletMambaBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 local_kernel_sizes=(3, 5)):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.local_mix = LocalMixing(dim, list(local_kernel_sizes), bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.four_direction_ssm = FourDirectionSSMMixing(dim, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.local_mix(self.norm1(x))
        x = x + self.four_direction_ssm(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x


class WaveletMambaFusionBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 local_kernel_sizes=(3, 5)):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.local_mix = LocalMixing(dim, list(local_kernel_sizes), bias)
        self.four_direction_ssm = FourDirectionSSMMixing(dim, bias)
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=True),
        )
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.residual_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        mixed_input = self.norm1(x)
        local_feat = self.local_mix(mixed_input)
        ssm_feat = self.four_direction_ssm(mixed_input)
        joint_feat = torch.cat([local_feat, ssm_feat], dim=1)
        local_gate, ssm_gate = torch.sigmoid(self.branch_gate(joint_feat)).chunk(2, dim=1)
        fused = self.fuse(torch.cat([local_feat * local_gate, ssm_feat * ssm_gate], dim=1))
        x = x + self.residual_scale * fused
        x = x + self.ffn_scale * self.ffn(self.norm2(x))
        return x


class WaveletMambaFusionAnchoredBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 local_kernel_sizes=(3, 5),
                 local_residual_ratio=0.25):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.local_mix = LocalMixing(dim, list(local_kernel_sizes), bias)
        self.four_direction_ssm = FourDirectionSSMMixing(dim, bias)
        self.local_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.residual_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_scale = nn.Parameter(torch.ones(1, dim, 1, 1))

        nn.init.zeros_(self.local_gate[-1].weight)
        nn.init.zeros_(self.local_gate[-1].bias)
        self._init_fuse_as_ssm_anchor(local_residual_ratio)

    def _init_fuse_as_ssm_anchor(self, local_residual_ratio):
        with torch.no_grad():
            self.fuse.weight.zero_()
            if self.fuse.bias is not None:
                self.fuse.bias.zero_()
            dim = self.fuse.out_channels
            for channel in range(dim):
                self.fuse.weight[channel, channel, 0, 0] = local_residual_ratio
                self.fuse.weight[channel, dim + channel, 0, 0] = 1.0

    def forward(self, x):
        mixed_input = self.norm1(x)
        local_feat = self.local_mix(mixed_input)
        ssm_feat = self.four_direction_ssm(mixed_input)
        local_gate = torch.sigmoid(self.local_gate(torch.cat([local_feat, ssm_feat], dim=1)))
        fused = self.fuse(torch.cat([local_feat * local_gate, ssm_feat], dim=1))
        x = x + self.residual_scale * fused
        x = x + self.ffn_scale * self.ffn(self.norm2(x))
        return x


class WaveletMambaFusionBiasedBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 local_kernel_sizes=(3, 5),
                 local_gate_bias=-1.0,
                 ssm_gate_bias=1.0):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.local_mix = LocalMixing(dim, list(local_kernel_sizes), bias)
        self.four_direction_ssm = FourDirectionSSMMixing(dim, bias)
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=True),
        )
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.residual_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_scale = nn.Parameter(torch.ones(1, dim, 1, 1))

        nn.init.zeros_(self.branch_gate[-1].weight)
        with torch.no_grad():
            self.branch_gate[-1].bias[:dim].fill_(local_gate_bias)
            self.branch_gate[-1].bias[dim:].fill_(ssm_gate_bias)

    def forward(self, x):
        mixed_input = self.norm1(x)
        local_feat = self.local_mix(mixed_input)
        ssm_feat = self.four_direction_ssm(mixed_input)
        joint_feat = torch.cat([local_feat, ssm_feat], dim=1)
        local_gate, ssm_gate = torch.sigmoid(self.branch_gate(joint_feat)).chunk(2, dim=1)
        fused = self.fuse(torch.cat([local_feat * local_gate, ssm_feat * ssm_gate], dim=1))
        x = x + self.residual_scale * fused
        x = x + self.ffn_scale * self.ffn(self.norm2(x))
        return x


class WaveletMambaModulatedBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 local_kernel_sizes=(3, 5)):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.local_mix = LocalMixing(dim, list(local_kernel_sizes), bias)
        self.local_res_scale = nn.Parameter(torch.ones(1, dim, 1, 1))

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ssm_modulation = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.ssm_modulation.weight)
        nn.init.zeros_(self.ssm_modulation.bias)
        self.ssm_modulation_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.four_direction_ssm = FourDirectionSSMMixing(dim, bias)
        self.ssm_res_scale = nn.Parameter(torch.ones(1, dim, 1, 1))

        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        local_feat = self.local_mix(self.norm1(x))
        x = x + self.local_res_scale * local_feat

        ssm_base = self.norm2(x)
        ssm_delta = torch.tanh(self.ssm_modulation(local_feat))
        ssm_input = ssm_base + self.ssm_modulation_scale * ssm_delta
        x = x + self.ssm_res_scale * self.four_direction_ssm(ssm_input)

        x = x + self.ffn_scale * self.ffn(self.norm3(x))
        return x


class WaveletMambaSpatialFusionBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 local_kernel_sizes=(3, 5)):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.local_mix = LocalMixing(dim, list(local_kernel_sizes), bias)
        self.four_direction_ssm = FourDirectionSSMMixing(dim, bias)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=True),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)

        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.residual_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        mixed_input = self.norm1(x)
        local_feat = self.local_mix(mixed_input)
        ssm_feat = self.four_direction_ssm(mixed_input)

        gate_logits = self.spatial_gate(torch.cat([local_feat, ssm_feat], dim=1))
        gate_logits = gate_logits.view(x.shape[0], 2, x.shape[1], x.shape[2], x.shape[3])
        branch_weights = torch.softmax(gate_logits, dim=1)
        local_weight = branch_weights[:, 0]
        ssm_weight = branch_weights[:, 1]

        fused = self.fuse(torch.cat([local_feat * local_weight, ssm_feat * ssm_weight], dim=1))
        x = x + self.residual_scale * fused
        x = x + self.ffn_scale * self.ffn(self.norm2(x))
        return x


class WaveletMambaResidualFusionBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 local_kernel_sizes=(3, 5)):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.local_mix = LocalMixing(dim, list(local_kernel_sizes), bias)
        self.four_direction_ssm = FourDirectionSSMMixing(dim, bias)

        self.local_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.local_gate[-1].weight)
        nn.init.zeros_(self.local_gate[-1].bias)

        self.ssm_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.local_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        mixed_input = self.norm1(x)
        local_feat = self.local_mix(mixed_input)
        ssm_feat = self.four_direction_ssm(mixed_input)
        local_weight = torch.sigmoid(self.local_gate(torch.cat([local_feat, ssm_feat], dim=1)))

        fused = self.ssm_scale * ssm_feat + self.local_scale * (local_weight * local_feat)
        x = x + fused
        x = x + self.ffn_scale * self.ffn(self.norm2(x))
        return x
