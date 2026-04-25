import torch
import torch.nn as nn
import ptwt
import pywt

try:
    from ptwt.wavelets_learnable import SoftOrthogonalWavelet, ProductFilter
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import learnable wavelets from ptwt. "
        "Please ensure `ptwt` is installed."
    ) from e


def _init_filters_from_pywt(init_wavelet: str):
    w = pywt.Wavelet(init_wavelet)
    dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
    dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)
    rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
    rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
    return dec_lo, dec_hi, rec_lo, rec_hi


def create_learnable_wavelet(init_wavelet: str = "db4", orthogonal: bool = True):
    """Create a learnable wavelet filter bank initialized from pywt."""
    dec_lo, dec_hi, rec_lo, rec_hi = _init_filters_from_pywt(init_wavelet)
    if orthogonal:
        return SoftOrthogonalWavelet(dec_lo, dec_hi, rec_lo, rec_hi)
    return ProductFilter(dec_lo, dec_hi, rec_lo, rec_hi)


class LearnableDWTForward(nn.Module):
    """1-level 2D DWT; applies ptwt per spatial channel, then stacks back to (B, C, ...).

    `ptwt.wavedec2` expects 1 input channel per batch row; we use (B*C, 1, H, W).

    Returns:
        yl: (B, C, H/2, W/2)
        yh: list with one tensor of shape (B, C, 3, H/2, W/2)
    """

    def __init__(self, wavelet, mode: str = "zero"):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = x.reshape(b * c, 1, h, w)
        coeffs = ptwt.wavedec2(x1, self.wavelet, level=1, mode=self.mode)
        yl = coeffs[0]
        lh, hl, hh = coeffs[1]
        _, _, h2, w2 = yl.shape
        yl = yl.view(b, c, h2, w2)
        lh = lh.view(b, c, h2, w2)
        hl = hl.view(b, c, h2, w2)
        hh = hh.view(b, c, h2, w2)
        yh0 = torch.stack([lh, hl, hh], dim=2)
        return yl, [yh0]


class LearnableDWTInverse(nn.Module):
    """1-level 2D IDWT; inverts the per-channel layout from `LearnableDWTForward`.

    `mode` is accepted for API parity with `LearnableDWTForward` (ptwt `waverec2` has no mode).
    """

    def __init__(self, wavelet, mode: str = "zero"):
        super().__init__()
        self.wavelet = wavelet
        # Stored for parity with LearnableDWTForward; ptwt.waverec2 ignores padding mode.
        self.mode = mode

    def forward(self, coeffs):
        yl, yh = coeffs
        yh0 = yh[0]
        b, c, _, h2, w2 = yh0.shape
        lh = yh0[:, :, 0, :, :]
        hl = yh0[:, :, 1, :, :]
        hh = yh0[:, :, 2, :, :]
        yl1 = yl.reshape(b * c, 1, yl.shape[2], yl.shape[3])
        lh1 = lh.reshape(b * c, 1, h2, w2)
        hl1 = hl.reshape(b * c, 1, h2, w2)
        hh1 = hh.reshape(b * c, 1, h2, w2)
        # ptwt: waverec2(coeffs, wavelet) only; boundary mode is fixed by wavedec2 coeffs
        out1 = ptwt.waverec2([yl1, (lh1, hl1, hh1)], self.wavelet)
        _, _, h_out, w_out = out1.shape
        return out1.view(b, c, h_out, w_out)

