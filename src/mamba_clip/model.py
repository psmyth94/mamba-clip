# type: ignore
import math
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import rearrange, repeat
from open_clip import (
    CustomTextCLIP,
    create_model_from_pretrained,
    get_tokenizer,
)
from open_clip.transform import PreprocessCfg
from open_clip_train.train import unwrap_model
from timm.layers.drop import DropPath
from torch.functional import Tensor
from torch.nn.init import trunc_normal_
from torch.utils import checkpoint
from transformers import PreTrainedModel

from .data import get_transform
from .utils import logging

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
    )
except ImportError:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except ImportError:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

logger = logging.get_logger(__name__)


def flops_selective_scan_ref(
    B=1,
    L=256,
    D=768,
    N=16,
    with_D=True,
    with_Z=False,
    with_Group=True,
    with_complex=False,
):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum(
            [[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln"
        )
    else:
        flops += get_flops_einsum(
            [[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln"
        )
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


class PatchEmbed2D(torch.nn.Module):
    r"""Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (torch.nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = torch.nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(torch.nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (torch.nn.Module, optional): Normalization layer.  Default: torch.nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = torch.nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(
                f"Warning, x.shape {x.shape} is not match even ===========", flush=True
            )
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x1 = x1[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x2 = x2[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x3 = x3[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(torch.nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = torch.nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        _, _, _, C = x.shape
        x = self.expand(x)

        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // self.dim_scale,
        )
        x = self.norm(x)

        return x


class Final_PatchExpand2D(torch.nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = torch.nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        _, _, _, C = x.shape
        x = self.expand(x)

        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // self.dim_scale,
        )
        x = self.norm(x)

        return x


class SS2D(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = torch.nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = torch.nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = torch.nn.SiLU()

        self.x_proj = (
            torch.nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            torch.nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            torch.nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            torch.nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = torch.nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = torch.nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = torch.nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=4, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = torch.nn.LayerNorm(self.d_inner)
        self.out_proj = torch.nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = torch.nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            torch.nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            torch.nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = torch.nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = torch.nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, _, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, _, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, height, width, groups, channels_per_group)

    x = torch.transpose(x, 3, 4).contiguous()

    # flatten
    x = x.view(batch_size, height, width, -1)

    return x


class SS_Conv_SSM(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            torch.nn.LayerNorm, eps=1e-6
        ),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim // 2)
        self.self_attention = SS2D(
            d_model=hidden_dim // 2, dropout=attn_drop_rate, d_state=d_state, **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.conv33conv33conv11 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(hidden_dim // 2),
            torch.nn.Conv2d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm2d(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm2d(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 2,
                kernel_size=1,
                stride=1,
            ),
            torch.nn.ReLU(),
        )
        # self.finalconv11 = torch.nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input: torch.Tensor):
        input_left, input_right = input.chunk(2, dim=-1)
        x = self.drop_path(self.self_attention(self.ln_1(input_right)))
        input_left = input_left.permute(0, 3, 1, 2).contiguous()
        input_left = self.conv33conv33conv11(input_left)
        input_left = input_left.permute(0, 2, 3, 1).contiguous()
        output = torch.cat((input_left, x), dim=-1)
        output = channel_shuffle(output, groups=2)
        return output + input


class VSSLayer(torch.nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (torch.nn.Module, optional): Normalization layer. Default: torch.nn.LayerNorm
        downsample (torch.nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=torch.nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = torch.nn.ModuleList(
            [
                SS_Conv_SSM(
                    hidden_dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )

        if True:  # is this really applied? Yes, but been overriden later in VSSM!

            def _init_weights(module: torch.nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(torch.nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (torch.nn.Module, optional): Normalization layer. Default: torch.nn.LayerNorm
        downsample (torch.nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=torch.nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = torch.nn.ModuleList(
            [
                SS_Conv_SSM(
                    hidden_dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )

        if True:  # is this really applied? Yes, but been overriden later in VSSM!

            def _init_weights(module: torch.nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSM(torch.nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 4, 2],
        depths_decoder=[2, 9, 2, 2],
        dims=[96, 192, 384, 768],
        dims_decoder=[768, 384, 192, 96],
        d_state=16,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=torch.nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2**i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = torch.nn.Parameter(
                torch.zeros(1, *self.patches_resolution, self.embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = torch.nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        dpr_decoder = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))
        ][::-1]

        self.layers = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6)
                if d_state is None
                else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # self.norm = norm_layer(self.num_features)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = (
            torch.nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else torch.nn.Identity()
        )

        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )

    def _init_weights(self, m: torch.nn.Module):
        """
        out_proj.weight which is previously initilized in SS_Conv_SSM, would be cleared in torch.nn.Linear
        no fc.weight found in the any of the model parameters
        no torch.nn.Embedding found in the any of the model parameters
        so the thing is, SS_Conv_SSM initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x


class ClipModel(torch.nn.Module):
    output_dict = torch.jit.Final[bool]

    def __init__(self, model: CustomTextCLIP):
        super().__init__()
        self.output_dict = True
        self.visual = model.visual
        self.text = model.text
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = model.logit_scale
        self.logit_bias = model.logit_bias

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text, secondary_text=None) -> dict:
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )

        if not hasattr(self.visual, "output_dim") and image_features is not None:
            self.visual.output_dim = image_features.shape[1:]
            if (
                isinstance(self.visual.output_dim, torch.Size)
                and len(self.visual.output_dim) == 1
            ):
                self.visual.output_dim = self.visual.output_dim[0]

        if not hasattr(self.text, "output_dim") and text_features is not None:
            self.text.output_dim = text_features.shape[1:]
            if (
                isinstance(self.text.output_dim, torch.Size)
                and len(self.text.output_dim) == 1
            ):
                self.text.output_dim = self.text.output_dim[0]

        secondary_text_features = None
        if secondary_text is not None:
            secondary_text_features = self.encode_text(secondary_text, normalize=True)

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if secondary_text is not None:
                out_dict["secondary_text_features"] = secondary_text_features
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict

        out = (image_features, text_features, self.logit_scale.exp())
        if secondary_text is not None:
            out += (secondary_text_features,)
        if self.logit_bias is not None:
            out += (self.logit_bias,)
        return out

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.text.transformer.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
                )
            return

        encoder = (
            self.text.transformer.encoder
            if hasattr(self.text.transformer, "encoder")
            else self.text.transformer
        )
        layer_list = getattr(encoder, "layer", getattr(encoder, "block", None))
        embeddings = getattr(
            self.text.transformer,
            "embeddings",
            getattr(self.text.transformer, "embed_tokens", None),
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
                )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T

        return image_logits, text_logits


class ClipClassifier(torch.nn.Module):
    def __init__(
        self,
        clip_model: ClipModel,
        feature_dim=None,
        num_classes: int = 2,
        use_visual_only=False,
        use_text_only=False,
        use_inner_prod=False,
    ):
        super().__init__()
        self.clip_model = unwrap_model(clip_model)
        self.num_classes = num_classes

        # Freeze the CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if feature_dim is None:
            image_feature_dim = getattr(
                self.clip_model.visual,
                "embed_dim",
                getattr(
                    self.clip_model.visual,
                    "output_dim",
                    getattr(self.clip_model.visual, "d_model", None),
                ),
            )
            text_feature_dim = getattr(
                self.clip_model.text,
                "embed_dim",
                getattr(
                    self.clip_model.text,
                    "output_dim",
                    getattr(self.clip_model.text, "d_model", None),
                ),
            )
            logger.info(
                f"Image feature dim: {image_feature_dim}, Text feature dim: {text_feature_dim}"
            )
            if text_feature_dim is None or image_feature_dim is None:
                raise ValueError(
                    "Could not find image and text feature dimensions in the model"
                )
            feature_dim = image_feature_dim + text_feature_dim
        self.use_visual_only = use_visual_only
        self.use_text_only = use_text_only
        self.use_inner_prod = use_inner_prod
        if use_visual_only or use_text_only or use_inner_prod:
            output_dim = feature_dim
        else:
            output_dim = feature_dim // 2
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, num_classes),
        )
        # self.fc = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, image, text):
        # Get CLIP features
        clip_output = self.clip_model(image, text)
        image_features = clip_output["image_features"]
        text_features = clip_output["text_features"]

        if self.use_visual_only:
            return self.fc(image_features)
        elif self.use_text_only:
            return self.fc(text_features)
        elif self.use_inner_prod:
            return self.fc(image_features * text_features)
        # Concatenate image and text features
        combined_features = torch.cat((image_features, text_features), dim=1)

        # Classification
        logits = self.fc(combined_features)

        return logits

    def get_logits(self, image_features, text_features):
        # compute the product of image and text features
        logits = image_features * text_features
        if self.clup_model.logit_bias is not None:
            logits += self.logit_bias
        return logits

    def classify(self, image, text):
        logits = self.forward(image, text)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities


class MambaVisionClassifier(torch.nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        num_classes: int = 2,
        dropout=0.1,
    ):
        super().__init__()
        self.config = unwrap_model(model).config
        self.model = unwrap_model(model).model
        self.num_classes = num_classes

        feature_dim = int(
            self.config.dim * 2 ** (len(self.config.depths) - 1)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feature_dim, num_classes),
        )

    def forward(self, image, *args, **kwargs):
        out = self.model.forward_features(image)
        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict):
            out = next(iter(out.values()))
        return self.fc(out)

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.model.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(self.model.levels[-unlocked_groups:])


def init_model(
    model,
    tokenizer=None,
    aug_cfg: Optional[Dict[str, Any]] = None,
    is_clip=False,
    use_tokenizer=False,
):
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    if model == "medmamba":
        model = VSSM(depths=[2, 2, 8, 2], dims=[64, 128, 256, 512], num_classes=2)
    elif isinstance(model, str):
        tokenizer = tokenizer or model
        model, _ = create_model_from_pretrained(f"hf-hub:{model}")
    elif callable(model):
        model = model()

    if is_clip:
        model = ClipModel(model)

    if use_tokenizer:
        if isinstance(tokenizer, str):
            tokenizer = get_tokenizer(f"hf-hub:{tokenizer}")
        elif callable(tokenizer):
            tokenizer = tokenizer()

    pp_cfg = None
    if hasattr(model, "visual") and hasattr(model.visual, "preprocess_cfg"):
        pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = get_transform(aug_cfg, pp_cfg, is_train=True)
    preprocess_val = get_transform(aug_cfg, pp_cfg, is_train=False)
    return model, preprocess_train, preprocess_val, tokenizer
