#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# Adapted from ml-mobileclip/mobileclip/models/mci.py for self-contained usage.
# Registers MCI0, MCI1, MCI2 (V1) and fastvit_mci3, fastvit_mci4 (V2) with timm.
import copy
from functools import partial
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models import register_model

from mobile_fetal_clip.models.mobileone import MobileOneBlock
from mobile_fetal_clip.models.replknet import ReparamLargeKernelConv

_MODELS_REGISTERED = False


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "fastvit_t": _cfg(crop_pct=0.9),
    "fastvit_s": _cfg(crop_pct=0.9),
    "fastvit_m": _cfg(crop_pct=0.95),
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False
) -> nn.Sequential:
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=1, groups=1,
            inference_mode=inference_mode, use_se=False, num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=1, groups=out_channels,
            inference_mode=inference_mode, use_se=False, num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0, groups=1,
            inference_mode=inference_mode, use_se=False, num_conv_branches=1,
        ),
    )


class MHSA(nn.Module):
    """Multi-headed Self Attention module."""

    def __init__(self, dim: int, head_dim: int = 32, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x


class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(self, patch_size: int, stride: int, in_channels: int,
                 embed_dim: int, inference_mode: bool = False, use_se: bool = False) -> None:
        super().__init__()
        block = [
            ReparamLargeKernelConv(
                in_channels=in_channels, out_channels=embed_dim,
                kernel_size=patch_size, stride=stride, groups=in_channels,
                small_kernel=3, inference_mode=inference_mode, use_se=use_se,
            ),
            MobileOneBlock(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=1, stride=1, padding=0, groups=1,
                inference_mode=inference_mode, use_se=False, num_conv_branches=1,
            ),
        ]
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RepMixer(nn.Module):
    """Reparameterizable token mixer."""

    def __init__(self, dim, kernel_size=3, use_layer_scale=True,
                 layer_scale_init_value=1e-5, inference_mode: bool = False):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim, out_channels=self.dim,
                kernel_size=self.kernel_size, stride=1,
                padding=self.kernel_size // 2, groups=self.dim, bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim, dim, kernel_size, padding=kernel_size // 2, groups=dim,
                use_act=False, use_scale_branch=False, num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim, dim, kernel_size, padding=kernel_size // 2, groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)
        if self.use_layer_scale:
            x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
        else:
            x = x + self.mixer(x) - self.norm(x)
        return x

    def reparameterize(self) -> None:
        if self.inference_mode:
            return
        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (self.mixer.id_tensor + self.mixer.reparam_conv.weight
                 - self.norm.reparam_conv.weight)
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim, out_channels=self.dim,
            kernel_size=self.kernel_size, stride=1,
            padding=self.kernel_size // 2, groups=self.dim, bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class ConvFFN(nn.Module):
    """Convolutional FFN Module."""

    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None,
                 out_channels: Optional[int] = None, act_layer: nn.Module = nn.GELU,
                 drop: float = 0.0) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, padding=3, groups=in_channels, bias=False),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepCPE(nn.Module):
    """Reparameterizable conditional positional encoding."""

    def __init__(self, in_channels: int, embed_dim: int = 768,
                 spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
                 inference_mode=False) -> None:
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple)
        assert len(spatial_shape) == 2

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.embed_dim,
                kernel_size=self.spatial_shape, stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim, bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels, embed_dim, spatial_shape, 1,
                int(spatial_shape[0] // 2), bias=True, groups=embed_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)
        return self.pe(x) + x

    def reparameterize(self) -> None:
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (self.in_channels, input_dim, self.spatial_shape[0], self.spatial_shape[1]),
            dtype=self.pe.weight.dtype, device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i, i % input_dim,
                self.spatial_shape[0] // 2, self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.embed_dim,
            kernel_size=self.spatial_shape, stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim, bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")


class RepMixerBlock(nn.Module):
    """Metaformer block with RepMixer as token mixer."""

    def __init__(self, dim: int, kernel_size: int = 3, mlp_ratio: float = 4.0,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.0,
                 drop_path: float = 0.0, use_layer_scale: bool = True,
                 layer_scale_init_value: float = 1e-5,
                 inference_mode: bool = False):
        super().__init__()
        self.token_mixer = RepMixer(
            dim, kernel_size=kernel_size, use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )
        assert mlp_ratio > 0
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim, hidden_channels=mlp_hidden_dim,
            act_layer=act_layer, drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class AttentionBlock(nn.Module):
    """Metaformer block with MHSA as token mixer."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.BatchNorm2d,
                 drop: float = 0.0, drop_path: float = 0.0,
                 use_layer_scale: bool = True,
                 layer_scale_init_value: float = 1e-5):
        super().__init__()
        self.norm = norm_layer(dim)
        self.token_mixer = MHSA(dim=dim)
        assert mlp_ratio > 0
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim, hidden_channels=mlp_hidden_dim,
            act_layer=act_layer, drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


def basic_blocks(
    dim: int, block_index: int, num_blocks: List[int], token_mixer_type: str,
    kernel_size: int = 3, mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0, drop_path_rate: float = 0.0,
    use_layer_scale: bool = True, layer_scale_init_value: float = 1e-5,
    inference_mode=False,
) -> nn.Sequential:
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "repmixer":
            blocks.append(RepMixerBlock(
                dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer, drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            ))
        elif token_mixer_type == "attention":
            blocks.append(AttentionBlock(
                dim, mlp_ratio=mlp_ratio, act_layer=act_layer,
                norm_layer=norm_layer, drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
        else:
            raise ValueError(f"Token mixer type: {token_mixer_type} not supported")
    return nn.Sequential(*blocks)


# ---------------------------------------------------------------------------
# FastViT backbone
# ---------------------------------------------------------------------------

class FastViT(nn.Module):
    """FastViT architecture from https://arxiv.org/pdf/2303.14189.pdf"""

    def __init__(
        self, layers, token_mixers: Tuple[str, ...], embed_dims=None,
        mlp_ratios=None, downsamples=None, se_downsamples=None,
        repmixer_kernel_size=3, norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU, num_classes=1000, pos_embs=None,
        down_patch_size=7, down_stride=2, drop_rate=0.0, drop_path_rate=0.0,
        use_layer_scale=True, layer_scale_init_value=1e-5, init_cfg=None,
        pretrained=None, cls_ratio=2.0, inference_mode=False, **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if pos_embs is None:
            pos_embs = [None] * len(layers)
        if se_downsamples is None:
            se_downsamples = [False] * len(layers)

        self.patch_embed = convolutional_stem(3, embed_dims[0], inference_mode)

        network = []
        for i in range(len(layers)):
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](embed_dims[i], embed_dims[i], inference_mode=inference_mode)
                )
            stage = basic_blocks(
                embed_dims[i], i, layers, token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size, mlp_ratio=mlp_ratios[i],
                act_layer=act_layer, norm_layer=norm_layer,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(PatchEmbed(
                    patch_size=down_patch_size, stride=down_stride,
                    in_channels=embed_dims[i], embed_dim=embed_dims[i + 1],
                    inference_mode=inference_mode, use_se=se_downsamples[i + 1],
                ))
        self.network = nn.ModuleList(network)

        self.num_features = int(embed_dims[-1] * cls_ratio)
        self.conv_exp = MobileOneBlock(
            in_channels=embed_dims[-1],
            out_channels=self.num_features,
            kernel_size=3, stride=1, padding=1, groups=embed_dims[-1],
            inference_mode=inference_mode, use_se=True, num_conv_branches=1,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0 else nn.Identity()
        )
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

    def cls_init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        """Reset the classifier head (timm interface)."""
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
        else:
            self.head = nn.Identity()

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(x)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled features before the head (timm interface)."""
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        cls_out = self.head(x)
        return cls_out


# ---------------------------------------------------------------------------
# V2 models (fastvit_mci3, fastvit_mci4) - uses timm's FastViT
# ---------------------------------------------------------------------------

def _create_fastvit_mci3_4():
    """Register fastvit_mci3 and fastvit_mci4 using timm's FastViT."""
    try:
        from timm.models.fastvit import _create_fastvit, RepConditionalPosEnc
        from timm.models.fastvit import MobileOneBlock as TimmMobileOneBlock

        class LayerNormChannel(nn.Module):
            """LayerNorm for Channel-first format 4D Tensor [B, C, H, W]."""
            def __init__(self, num_features, eps=1e-05) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(num_features))
                self.bias = nn.Parameter(torch.zeros(num_features))
                self.eps = eps

            def forward(self, x) -> torch.Tensor:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
                    + self.bias.unsqueeze(-1).unsqueeze(-1)
                return x

        def convolutional_stem_timm(in_chs, out_chs, act_layer=nn.GELU,
                                     inference_mode=False):
            return nn.Sequential(
                TimmMobileOneBlock(
                    in_chs=in_chs, out_chs=out_chs, kernel_size=3, stride=2,
                    act_layer=act_layer, inference_mode=inference_mode,
                    use_scale_branch=False,
                ),
                TimmMobileOneBlock(
                    in_chs=out_chs, out_chs=out_chs, kernel_size=3, stride=2,
                    group_size=1, act_layer=act_layer,
                    inference_mode=inference_mode, use_scale_branch=False,
                ),
                TimmMobileOneBlock(
                    in_chs=out_chs, out_chs=out_chs, kernel_size=1, stride=1,
                    act_layer=act_layer, inference_mode=inference_mode,
                    use_scale_branch=False,
                ),
            )

        v2_cfg = _cfg()

        @register_model
        def fastvit_mci3(pretrained=False, **kwargs):
            updated_kwargs = {
                'num_classes': kwargs.get('num_classes'),
                'global_pool': kwargs.get('global_pool'),
                'drop_path_rate': kwargs.get('drop_path_rate'),
            }
            model_args = dict(
                layers=(2, 12, 24, 4, 2),
                embed_dims=(96, 192, 384, 768, 1536),
                mlp_ratios=(4, 4, 4, 4, 4),
                se_downsamples=(False, False, False, False, False),
                downsamples=(False, True, True, True, True),
                pos_embs=(None, None, None,
                          partial(RepConditionalPosEnc, spatial_shape=(7, 7)),
                          partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
                token_mixers=("repmixer", "repmixer", "repmixer", "attention", "attention"),
                lkc_use_act=True,
                norm_layer=LayerNormChannel,
            )
            stem = convolutional_stem_timm(3, model_args['embed_dims'][0], nn.GELU, False)
            model = _create_fastvit(
                'fastvit_mci3', pretrained=pretrained,
                pretrained_cfg=v2_cfg, **dict(model_args, **updated_kwargs)
            )
            model.stem = stem
            return model

        @register_model
        def fastvit_mci4(pretrained=False, **kwargs):
            updated_kwargs = {
                'num_classes': kwargs.get('num_classes'),
                'global_pool': kwargs.get('global_pool'),
                'drop_path_rate': kwargs.get('drop_path_rate'),
            }
            model_args = dict(
                layers=(2, 12, 24, 4, 4),
                embed_dims=(128, 256, 512, 1024, 2048),
                mlp_ratios=(4, 4, 4, 4, 4),
                se_downsamples=(False, False, False, False, False),
                downsamples=(False, True, True, True, True),
                pos_embs=(None, None, None,
                          partial(RepConditionalPosEnc, spatial_shape=(7, 7)),
                          partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
                token_mixers=("repmixer", "repmixer", "repmixer", "attention", "attention"),
                lkc_use_act=True,
                norm_layer=LayerNormChannel,
            )
            stem = convolutional_stem_timm(3, model_args['embed_dims'][0], nn.GELU, False)
            model = _create_fastvit(
                'fastvit_mci4', pretrained=pretrained,
                pretrained_cfg=v2_cfg, **dict(model_args, **updated_kwargs)
            )
            model.stem = stem
            return model

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# V1 model registrations (mci0, mci1, mci2)
# ---------------------------------------------------------------------------

@register_model
def mci0(pretrained=False, **kwargs):
    """MCi0 — smallest variant (11.4M image params)."""
    layers = [2, 6, 10, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers, token_mixers=token_mixers, embed_dims=embed_dims,
        pos_embs=pos_embs, mlp_ratios=mlp_ratios,
        downsamples=downsamples, se_downsamples=se_downsamples, **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    return model


@register_model
def mci1(pretrained=False, **kwargs):
    """MCi1 — medium variant."""
    layers = [4, 12, 20, 4]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers, token_mixers=token_mixers, embed_dims=embed_dims,
        pos_embs=pos_embs, mlp_ratios=mlp_ratios,
        downsamples=downsamples, se_downsamples=se_downsamples, **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    return model


@register_model
def mci2(pretrained=False, **kwargs):
    """MCi2 — larger variant (35.7M image params)."""
    layers = [4, 12, 24, 4]
    embed_dims = [80, 160, 320, 640]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers, token_mixers=token_mixers, embed_dims=embed_dims,
        pos_embs=pos_embs, mlp_ratios=mlp_ratios,
        downsamples=downsamples, se_downsamples=se_downsamples, **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    return model


# Also register V1 models under fastvit_mci* names for MobileCLIP2 config compatibility
@register_model
def fastvit_mci0(pretrained=False, **kwargs):
    """Alias for mci0 under fastvit_mci0 name."""
    return mci0(pretrained=pretrained, **kwargs)


@register_model
def fastvit_mci2(pretrained=False, **kwargs):
    """Alias for mci2 under fastvit_mci2 name."""
    return mci2(pretrained=pretrained, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_all_mci_models():
    """Register all MCI models with timm. Safe to call multiple times."""
    global _MODELS_REGISTERED
    if _MODELS_REGISTERED:
        return
    _MODELS_REGISTERED = True
    # V1 models (mci0, mci1, mci2, fastvit_mci0, fastvit_mci2) are registered
    # via @register_model decorators at module level.
    # V2 models (fastvit_mci3, fastvit_mci4) need timm's fastvit module:
    _create_fastvit_mci3_4()
