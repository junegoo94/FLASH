# flash_improved.py
#
# FLASH (model6.py) 에서 세 가지 edge blurring 버그를 수정한 버전.
#
# Fix 1 [FA module] : mean(X, dim=C) → channel-wise FFT
#   - 기존: 96채널을 평균 내어 1채널로 줄인 뒤 FFT
#   - 개선: 모든 채널에 독립적으로 FFT 적용 → edge 정보 보존
#   - 근거: GFNet (Rao et al., NeurIPS 2021)
#
# Fix 2 [FA module] : scalar α → spatial α_map
#   - 기존: 공간 전체에 동일한 α 스칼라로 frequency branch 혼합
#   - 개선: Conv1x1으로 위치별 α_map 생성 → edge 구간에서 자동으로 높아짐
#   - 근거: CBAM (Woo et al., ECCV 2018) - 이미 FLASH에서 사용 중인 아이디어
#
# Fix 3 [MSF module] : Xe + Xd → gated fusion
#   - 기존: encoder/decoder feature를 단순 합산 → destructive interference
#   - 개선: gate = sigmoid(Conv1x1(cat(Xe, Xd))) 로 위치별 비율 결정
#   - 근거: SFNet (Li et al., CVPR 2022)
#
# ── Ablation 지원 ──────────────────────────────────────────────
#   use_fix1 / use_fix2 / use_fix3 flag로 각 Fix를 독립적으로 on/off.
#
#   ★ Checkpoint 호환 보장 ★
#     __init__ 에서 모든 모듈(freq_conv, alpha_gate, GatedMSF, freq_weight)을
#     항상 생성하므로 state_dict key가 flag와 무관하게 동일.
#     → 기존 checkpoint (use_fix1=True, use_fix2=True, use_fix3=True) 로드 후
#       eval 시 동일 flag 주면 완벽 재현.
#     → ablation용 새 모델은 flag 조합만 바꿔서 처음부터 학습.
#
# TULIP engine_upsampling.py 호환:
#   model(images_low_res, images_high_res, eval=True/False) 시그니처 유지
#   반환: (pred, total_loss, pixel_loss)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import collections.abc

# TULIP 원본 컴포넌트 재사용
from model.tulip import (
    PatchEmbedding, PatchMerging, PatchExpanding,
    FinalPatchExpanding, PixelShuffleHead,
    DropPath, Mlp,
)


# ─────────────────────────────────────────────────────────────
# FrequencyAwareWindowAttention  (Fix 1 + Fix 2 ablation 지원)
# ─────────────────────────────────────────────────────────────

class FrequencyAwareWindowAttention(nn.Module):
    """
    FA module.  use_fix1 / use_fix2 flag로 ablation 가능.

    ★ 모든 모듈은 flag 와 무관하게 항상 __init__ 에서 생성 ★
      → 기존 checkpoint 로드 시 state_dict key 불일치 없음.

    use_fix1=False: 원본 FLASH 방식 (channel mean → FFT)
    use_fix1=True:  channel-wise FFT (Fix 1)

    use_fix2=False: 원본 FLASH 방식 (scalar freq_weight)
    use_fix2=True:  spatial α_map (Fix 2)
    """

    def __init__(self, dim: int, window_size, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.,
                 proj_drop: float = 0., shift: bool = False,
                 fft_normalize: bool = False,
                 use_fix1: bool = True,
                 use_fix2: bool = True):
        super().__init__()
        self.dim          = dim
        self.window_size  = (
            window_size if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )
        self.num_heads    = num_heads
        self.scale        = (dim // num_heads) ** -0.5
        self.shift        = shift
        self.fft_normalize = fft_normalize
        self.use_fix1     = use_fix1
        self.use_fix2     = use_fix2

        self.num_windows       = self.window_size[0] * self.window_size[1]
        self.backup_window_size = (1, self.num_windows)
        self.backup_shift_size  = (0, self.num_windows // 2)
        self.shift_size = (
            (self.window_size[0] // 2, self.window_size[1] // 2) if shift else (0, 0)
        )

        # ── Spatial attention (TULIP 원본) ──────────────────────
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

        # ── Fix 1: channel-wise frequency conv ──────────────────
        # 항상 생성 (checkpoint 호환)
        # use_fix1=False 일 때는 원본 FLASH의 1채널 conv 로 대체
        self.freq_conv = nn.Sequential(          # Fix1 용 (C→C depthwise)
            nn.Conv2d(dim, dim, 1, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.freq_conv_orig = nn.Sequential(     # 원본 FLASH 용 (1→1)
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )

        # ── Fix 2: spatial α_map ────────────────────────────────
        # 항상 생성 (checkpoint 호환)
        self.alpha_gate   = nn.Sequential(       # Fix2 용 (C→1 spatial map)
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )
        self.freq_weight  = nn.Parameter(        # 원본 FLASH 용 scalar α
            torch.tensor(0.1)
        )

        self._init_relative_position_index()

    # ── 내부 유틸 ────────────────────────────────────────────────

    def _init_relative_position_index(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

    def window_partition(self, x):
        _, H, W, _ = x.shape
        return rearrange(
            x, 'B (Nh Mh) (Nw Mw) C -> (B Nh Nw) Mh Mw C',
            Nh=H // self.window_size[0], Nw=W // self.window_size[1]
        )

    def window_reverse(self, windows, H, W):
        return rearrange(
            windows, '(B Nh Nw) Mh Mw C -> B (Nh Mh) (Nw Mw) C',
            Nh=H // self.window_size[0], Nw=W // self.window_size[1]
        )

    def create_mask(self, x):
        _, H, W, _ = x.shape
        img_mask = torch.zeros((1, H, W, 1), device=x.device)
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = self.window_partition(img_mask)
        mask_windows = mask_windows.contiguous().view(-1, self.num_windows)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    # ── Forward ─────────────────────────────────────────────────

    def forward(self, x):
        B, H, W, C = x.shape

        if H < self.window_size[0]:
            self.window_size = self.backup_window_size
            if self.shift:
                self.shift_size = self.backup_shift_size

        # ── Spatial attention branch (원본 그대로) ───────────────
        if self.shift:
            x_shifted = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            mask = self.create_mask(x_shifted)
        else:
            x_shifted, mask = x, None

        x_windows = self.window_partition(x_shifted)
        Bn, Mh, Mw, _ = x_windows.shape
        x_flat = rearrange(x_windows, 'Bn Mh Mw C -> Bn (Mh Mw) C')

        qkv = rearrange(self.qkv(x_flat), 'Bn L (T Nh P) -> T Bn Nh L P', T=3, Nh=self.num_heads)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.num_windows, self.num_windows, -1).permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.num_heads, Mh * Mw, Mh * Mw) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Mh * Mw, Mh * Mw)

        attn = self.attn_drop(self.softmax(attn))
        x_spatial = rearrange(attn @ v, 'Bn Nh (Mh Mw) P -> Bn Mh Mw (Nh P)', Mh=Mh)
        x_spatial = self.proj_drop(self.proj(x_spatial))
        x_spatial = self.window_reverse(x_spatial, H, W)
        if self.shift:
            x_spatial = torch.roll(x_spatial, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        # ── Frequency branch ─────────────────────────────────────
        x_cf = x.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)

        with torch.cuda.amp.autocast(enabled=False):
            x_cf32 = x_cf.float()

            if self.use_fix1:
                # Fix 1: channel-wise FFT
                x_fft = torch.fft.rfft2(x_cf32, dim=(-2, -1))
                if self.fft_normalize:
                    x_fft_abs = x_fft.abs() / float(H * W)
                else:
                    x_fft_abs = x_fft.abs()
                freq_attn   = self.freq_conv(x_fft_abs)       # (B, C, H, W//2+1)
                x_freq_out  = torch.fft.irfft2(x_fft * freq_attn, s=(H, W), dim=(-2, -1))
            else:
                # 원본 FLASH: channel mean → FFT
                x_mean = x_cf32.mean(dim=1, keepdim=True)     # (B, 1, H, W)
                x_fft  = torch.fft.rfft2(x_mean, dim=(-2, -1))
                freq_attn  = self.freq_conv_orig(x_fft.abs())  # (B, 1, H, W//2+1)
                x_freq_out = torch.fft.irfft2(
                    x_fft * freq_attn, s=(H, W), dim=(-2, -1)
                ).expand(-1, C, -1, -1)                        # (B, C, H, W)

            if self.use_fix2:
                # Fix 2: spatial α_map
                alpha = self.alpha_gate(x_cf32)                # (B, 1, H, W)
            else:
                # 원본 FLASH: scalar α
                alpha = self.freq_weight                        # scalar

        x_freq_out = x_freq_out.to(x_spatial.dtype).permute(0, 2, 3, 1)  # (B,H,W,C)

        if self.use_fix2:
            alpha = alpha.to(x_spatial.dtype).permute(0, 2, 3, 1)         # (B,H,W,1)
        # else: alpha is scalar, broadcasts fine

        return x_spatial + alpha * x_freq_out


# ─────────────────────────────────────────────────────────────
# GatedMSF  (Fix 3 ablation 지원)
# ─────────────────────────────────────────────────────────────

class GatedMSF(nn.Module):
    """
    MSF module.  use_fix3 flag로 ablation 가능.

    ★ gate_conv 는 항상 생성 ★  → checkpoint 호환
    use_fix3=False: 원본 FLASH 덧셈 (Xcombined = Xe + Xd)
    use_fix3=True:  gated fusion (Fix 3)
    """

    def __init__(self, in_dim: int, out_dim: int, use_fix3: bool = True):
        super().__init__()
        self.use_fix3 = use_fix3
        self.align    = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # Fix 3: gate (항상 생성)
        self.gate_conv = nn.Conv2d(out_dim * 2, out_dim, 1)

        # Multi-scale convolutions (원본 동일)
        self.conv1x1 = nn.Conv2d(out_dim, out_dim, 1)
        self.conv3x3 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv5x5 = nn.Conv2d(out_dim, out_dim, 5, padding=2)

        self.fusion_weights = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, 1),
            nn.ReLU(),
            nn.Conv2d(out_dim, 3, 1),
            nn.Softmax(dim=1)
        )

        self.channel_attn = ChannelAttention(out_dim)
        self.spatial_attn = SpatialAttention()

        self.project = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.ReLU()
        )

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        enc    = self.align(enc)                              # (B,H,W,C)
        enc_cf = enc.permute(0, 3, 1, 2)                     # (B,C,H,W)
        dec_cf = dec.permute(0, 3, 1, 2)

        if self.use_fix3:
            # Fix 3: gated fusion
            gate       = torch.sigmoid(self.gate_conv(torch.cat([enc_cf, dec_cf], dim=1)))
            combined   = gate * enc_cf + (1.0 - gate) * dec_cf
        else:
            # 원본 FLASH: 단순 덧셈
            combined   = enc_cf + dec_cf

        f1 = self.conv1x1(combined)
        f3 = self.conv3x3(combined)
        f5 = self.conv5x5(combined)

        w      = self.fusion_weights(torch.cat([f1, f3, f5], dim=1))
        fused  = w[:, 0:1] * f1 + w[:, 1:2] * f3 + w[:, 2:3] * f5
        fused  = self.channel_attn(fused) * fused
        fused  = self.spatial_attn(fused) * fused
        return self.project(fused).permute(0, 2, 3, 1)       # (B,H,W,C)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


# ─────────────────────────────────────────────────────────────
# EnhancedSwinBlock
# ─────────────────────────────────────────────────────────────

class EnhancedSwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 8),
                 shift=False, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fft_normalize=False,
                 use_fix1=True, use_fix2=True):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.norm2     = norm_layer(dim)
        self.attn      = FrequencyAwareWindowAttention(
            dim=dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, shift=shift,
            fft_normalize=fft_normalize,
            use_fix1=use_fix1, use_fix2=use_fix2
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp       = Mlp(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────
# Encoder / Decoder
# ─────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, index, depths, embed_dim, num_heads, window_size,
                 mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                 drop_path_rates, norm_layer, downsample,
                 fft_normalize=False, use_fix1=True, use_fix2=True):
        super().__init__()
        dim      = embed_dim * 2 ** index
        self.blocks = nn.ModuleList([
            EnhancedSwinBlock(
                dim=dim, num_heads=num_heads[index], window_size=window_size,
                shift=(i % 2 == 1), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[i], norm_layer=norm_layer,
                fft_normalize=fft_normalize,
                use_fix1=use_fix1, use_fix2=use_fix2
            ) for i in range(depths[index])
        ])
        self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, index, depths, embed_dim, num_heads, window_size,
                 mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                 drop_path_rates, norm_layer, upsample,
                 fft_normalize=False, use_fix1=True, use_fix2=True):
        super().__init__()
        rev_idx = len(depths) - index - 2
        dim     = embed_dim * 2 ** rev_idx
        self.blocks = nn.ModuleList([
            EnhancedSwinBlock(
                dim=dim, num_heads=num_heads[rev_idx], window_size=window_size,
                shift=(i % 2 == 1), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[i], norm_layer=norm_layer,
                fft_normalize=fft_normalize,
                use_fix1=use_fix1, use_fix2=use_fix2
            ) for i in range(depths[rev_idx])
        ])
        self.upsample = PatchExpanding(dim=dim, norm_layer=norm_layer) if upsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.upsample(x)


# ─────────────────────────────────────────────────────────────
# FLASHImproved — 메인 모델
# ─────────────────────────────────────────────────────────────

class FLASHImproved(nn.Module):
    """
    FLASH + Fix 1/2/3.  각 Fix 는 use_fix1/2/3 flag 로 독립 제어.

    ★ Checkpoint 호환 ★
      모든 서브모듈은 flag 무관하게 항상 생성됨.
      기존 checkpoint (all fixes ON) 로드 후 동일 flag 로 eval 하면 완벽 재현.
      ablation 실험은 flag 조합을 바꿔 처음부터 학습.

    TULIP engine_upsampling.py 호환:
      forward(x, target, eval=False) → (pred, total_loss, pixel_loss)
    """

    def __init__(self,
                 img_size=(16, 1024),
                 target_img_size=(64, 1024),
                 patch_size=(1, 4),
                 in_chans=1,
                 embed_dim=96,
                 window_size=(2, 8),
                 depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 circular_padding=True,
                 pixel_shuffle=True,
                 log_transform=True,
                 fft_normalize=False,
                 use_fix1=True,
                 use_fix2=True,
                 use_fix3=True):
        super().__init__()
        self.log_transform  = log_transform
        self.fft_normalize  = fft_normalize
        self.use_fix1       = use_fix1
        self.use_fix2       = use_fix2
        self.use_fix3       = use_fix3
        self.num_layers     = len(depths)
        self.embed_dim      = embed_dim

        # ── Patch embedding ──────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            circular_padding=circular_padding
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ── Encoder ─────────────────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                index=i, depths=depths, embed_dim=embed_dim, num_heads=num_heads,
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rates=dpr[sum(depths[:i]):sum(depths[:i+1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_layers - 1),
                fft_normalize=fft_normalize,
                use_fix1=use_fix1, use_fix2=use_fix2
            ) for i in range(self.num_layers)
        ])

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = PatchExpanding(
            dim=embed_dim * 2 ** (self.num_layers - 1), norm_layer=norm_layer
        )

        # ── Decoder ─────────────────────────────────────────────
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                index=i, depths=depths, embed_dim=embed_dim, num_heads=num_heads,
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rates=dpr[sum(depths[:self.num_layers-2-i]):
                                    sum(depths[:self.num_layers-1-i])],
                norm_layer=norm_layer,
                upsample=(i < self.num_layers - 2),
                fft_normalize=fft_normalize,
                use_fix1=use_fix1, use_fix2=use_fix2
            ) for i in range(self.num_layers - 1)
        ])

        # ── Skip connections: GatedMSF (항상 생성, use_fix3로 동작 제어) ──
        self.skip_connections = nn.ModuleList([
            GatedMSF(
                in_dim=embed_dim * 2 ** (self.num_layers - 2 - i),
                out_dim=embed_dim * 2 ** (self.num_layers - 2 - i),
                use_fix3=use_fix3
            ) for i in range(self.num_layers - 1)
        ])

        # ── Output head ──────────────────────────────────────────
        self.norm_up = norm_layer(embed_dim)
        upscale = int(
            ((target_img_size[0] * target_img_size[1]) /
             (img_size[0] * img_size[1])) ** 0.5
        ) * 2 * int(((patch_size[0] * patch_size[1]) // 4) ** 0.5)

        self.pixel_shuffle_flag = pixel_shuffle
        if pixel_shuffle:
            self.ps_head      = PixelShuffleHead(dim=embed_dim, upscale_factor=upscale)
            self.decoder_pred = nn.Conv2d(embed_dim, in_chans, 1, bias=False)
        else:
            self.final_expand = FinalPatchExpanding(
                dim=embed_dim, norm_layer=norm_layer, upscale_factor=upscale
            )
            self.decoder_pred = nn.Conv2d(
                embed_dim // (upscale ** 2), in_chans, 1, bias=False
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.pos_drop(self.patch_embed(x))   # (B,H,W,C)
        enc_feats = []
        for layer in self.encoder_layers:
            enc_feats.append(x)
            x = layer(x)
        x = self.bottleneck(x)
        for i, layer in enumerate(self.decoder_layers):
            x = self.skip_connections[i](enc_feats[-(i + 2)], x)
            x = layer(x)
        return self.norm_up(x)

    def forward(self, x, target, eval=False, mc_drop=False):
        feats = self.forward_features(x)
        if self.pixel_shuffle_flag:
            pred = self.decoder_pred(self.ps_head(rearrange(feats, 'B H W C -> B C H W').contiguous()))
        else:
            pred = self.decoder_pred(rearrange(self.final_expand(feats), 'B H W C -> B C H W').contiguous())

        total_loss, pixel_loss = self._compute_loss(pred, target)
        return pred, total_loss, pixel_loss

    def _compute_loss(self, pred, target):
        loss = (pred - target).abs().mean()
        pixel_loss = (
            (torch.expm1(pred) - torch.expm1(target)).abs().mean()
            if self.log_transform else loss.clone()
        )
        return loss, pixel_loss


# ─────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────

def flash_improved_base(**kwargs):
    return FLASHImproved(depths=(2,2,2,2), embed_dim=96,
                         num_heads=(3,6,12,24), **kwargs)

def flash_improved_large(**kwargs):
    return FLASHImproved(depths=(2,2,2,2,2), embed_dim=96,
                         num_heads=(3,6,12,24,48), **kwargs)