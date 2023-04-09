# =========================================
# Vision Transformer Network
# =========================================
import math
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.runner import BaseModule, ModuleList, Sequential
from mit.embed import PatchEmbed


def resize(input_,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input_.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input_ size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input_, size, scale_factor, mode, align_corners)

def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input_ tensor of shape [N, L, C] before convertion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after convertion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input_ tensor of shape [N, C, H, W] before convertion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after convertion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()





class MixFFN(BaseModule, ABC):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=None,
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None
                 ):
        super(MixFFN, self).__init__(init_cfg)

        if act_cfg is None:
            act_cfg = dict(type='GELU')
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)

        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)

        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)

        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention, ABC):
    """An implementation of Efficient Multi-head Attention of Segformer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=None,
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # `need_weights=True` will let nn.MultiHeadAttention
        # `return attn_output, attn_output_weights.sum(dim=1) / num_heads`
        # The `attn_output_weights.sum(dim=1)` may cause cuda error. So, we set
        # `need_weights=False` to ignore `attn_output_weights.sum(dim=1)`.
        # This issue - `https://github.com/pytorch/pytorch/issues/37583` report
        # the error that large scale tensor sum operation may cause cuda error.

        #---------------------------------------------------#
        #  这里需要交换第0维度和第1维度的顺序
        #---------------------------------------------------#
        # out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]
        out = self.attn(query=x_q.transpose(0, 1), key=x_kv.transpose(0, 1), value=x_kv.transpose(0, 1), need_weights=False)[0]
        return identity + self.dropout_layer(self.proj_drop(out.transpose(0,1)))

class TransformerEncoderLayer(BaseModule, ABC):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Defalut: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=None,
                 norm_cfg=None,
                 batch_first=False, # 这里需要修改为false
                 sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        if act_cfg is None:
            act_cfg = dict(type='GELU')

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x


class MixVisionTransformer(BaseModule, ABC):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.

    Args:
        in_channels (int): Number of input_ channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stages (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Defalut: dict(type='GELU').
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=None,
                 num_heads=None,
                 patch_sizes=None,
                 strides=None,
                 sr_ratios=None,
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=None,
                 norm_cfg=None
                 ):
        super().__init__()
        if strides is None:
            strides = [4, 2, 2, 2]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if patch_sizes is None:
            patch_sizes = [7, 3, 3, 3]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if num_layers is None:
            num_layers = [3, 4, 6, 3]
        if act_cfg is None:
            act_cfg = dict(type='GELU')
        if norm_cfg is None:
            norm_cfg = dict(type='LN', eps=1e-6)

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        assert num_stages == len(num_layers) == len(num_heads) \
            == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                pad_to_patch_size=False,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels
                fan_out //= m.groups
                normal_init(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    constant_init(m.bias, 0)

    def forward(self, x):
        outs = []

        for i, layer in enumerate(self.layers):
            x, H, W = layer[0](x), layer[0].DH, layer[0].DW
            hw_shape = (H, W)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs


class FuseNeck(nn.Module, ABC):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,
                 interpolate_mode='bilinear',
                 in_channels=None,
                 in_index=None,
                 channels = 256,
                 act_cfg=None,
                 norm_cfg=None,
                 align_corners = False,
                 neck_scale = 1
                 ):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if in_index is None:
            in_index = [0, 1, 2, 3]
        if in_channels is None:
            in_channels = [64, 128, 320, 512]

        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners
        self.neck_scale = neck_scale
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input_=conv(x),
                    size=inputs[self.neck_scale].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    # =========================================
    # 参数加载
    # =========================================
    import yaml
    CFG_PATH = "../cfgs/ultranet_mit_b1.yml"
    cfgs = yaml.safe_load(open(CFG_PATH))

    # input_ parameters
    batch_size = 6
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    dummy_input = torch.randn((batch_size, 3, net_input_height, net_input_width)).to("cuda:0")

    # backbone paramters
    in_channels = cfgs["backbone"]["in_channels"]
    embed_dims = cfgs["backbone"]["embed_dims"]
    num_stages = cfgs["backbone"]["num_stages"]
    num_layers = cfgs["backbone"]["num_layers"]
    num_heads = cfgs["backbone"]["num_heads"]
    patch_sizes = cfgs["backbone"]["patch_sizes"]
    strides = cfgs["backbone"]["strides"]
    sr_ratios = cfgs["backbone"]["sr_ratios"]
    out_indices = tuple(cfgs["backbone"]["out_indices"])
    mlp_ratio = cfgs["backbone"]["mlp_ratio"]
    qkv_bias = cfgs["backbone"]["qkv_bias"]
    drop_rate = cfgs["backbone"]["drop_rate"]
    attn_drop_rate = cfgs["backbone"]["attn_drop_rate"]
    drop_path_rate = cfgs["backbone"]["drop_path_rate"]
    act_cfg = cfgs["backbone"]["act_cfg"]
    norm_cfg = cfgs["backbone"]["norm_cfg"]

    # neck parameters
    interpolate_mode = cfgs["backbone"]["interpolate_mode"]
    in_channels_ = cfgs["backbone"]["in_channels_"]
    in_index = cfgs["backbone"]["in_index"]
    channels = cfgs["backbone"]["channels"]
    act_cfg_ = cfgs["backbone"]["act_cfg_"]
    norm_cfg_ = cfgs["backbone"]["norm_cfg_"]
    align_corners = cfgs["backbone"]["align_corners"]

    # =========================================
    # 构建模型
    # =========================================
    backbone = MixVisionTransformer(
                 in_channels=in_channels,
                 embed_dims=embed_dims,
                 num_stages=num_stages,
                 num_layers=num_layers,
                 num_heads=num_heads,
                 patch_sizes=patch_sizes,
                 strides=strides,
                 sr_ratios=sr_ratios,
                 out_indices=out_indices,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate,
                 act_cfg=act_cfg,
                 norm_cfg=norm_cfg,
    ).cuda()


    neck = FuseNeck(
                     interpolate_mode=interpolate_mode,
                     in_channels=in_channels_,
                     in_index = in_index,
                     channels = channels,
                     act_cfg=act_cfg_,
                     norm_cfg = norm_cfg_,
                     align_corners = align_corners,
                 ).cuda()

    # =========================================
    # 测试模型
    # =========================================
    feats = None
    fused_feats = None
    import time
    for _ in range(100):
        tic = time.time()
        torch.cuda.synchronize()
        feats = backbone(dummy_input)
        fused_feats = neck(feats)
        torch.cuda.synchronize()
        print("inference time is: %f"  %(1000*(time.time() - tic)))

    print("multi level feature from transformer")
    for mvl_feat in feats:
        print(mvl_feat.shape)

    print("fused feature from neck")
    print(fused_feats.shape)