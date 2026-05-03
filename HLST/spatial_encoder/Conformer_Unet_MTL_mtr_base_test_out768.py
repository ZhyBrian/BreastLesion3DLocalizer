import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as cp
import math

from typing import Any, List, Tuple
from collections import OrderedDict
from torch import Tensor
from torchvision.utils import save_image

from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, 1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = nn.LayerNorm(dim, 1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(med_planes, eps=1e-6)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(med_planes, eps=1e-6)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes, eps=1e-6)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = nn.BatchNorm2d(outplanes, eps=1e-6)

        self.res_conv = res_conv

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = nn.LayerNorm(outplanes, 1e-6)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(outplanes, 1e-6)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class ConvTransBlock(nn.Module):

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, groups=1):
        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups)
        self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)
        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)
        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class Conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale)

        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale
                            )
                            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        init_stage = fin_stage
        fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale
                            )
                            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        init_stage = fin_stage
        fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale
                            )
                            )
        self.fin_stage = fin_stage

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x_out_1 = self.act1(self.bn1(self.conv1(x)))
        x_base = self.maxpool(x_out_1)

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)
            if i == 4:
                x_out_2 = x
            elif i == 8:
                x_out_3 = x
            elif i == 12:
                x_out_4 = x

        return x_out_4, x_out_3, x_out_2, x_out_1, x_t


class BasicConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(BasicConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, conv_num=2, intermediate_out=False):
        super(MultiConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.intermediate_out = intermediate_out
        assert conv_num >= 2

        self.conv1 = BasicConv(in_channels, mid_channels)
        conv_list = []
        for i in range(conv_num - 1):
            if i != conv_num - 2:
                conv_list.append(BasicConv(mid_channels, mid_channels))
            else:
                conv_list.append(BasicConv(mid_channels, out_channels))
        self.conv2 = nn.Sequential(*conv_list)

    def forward(self, x):
        x_mid = self.conv1(x)
        x_out = self.conv2(x_mid)

        if self.intermediate_out:
            return x_out, x_mid
        else:
            return x_out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, conv_num=2, skip=True):
        super(Up, self).__init__()
        self.skip = skip
        if self.skip:
            self.mid_channels = out_channels
            self.skip_channels = out_channels
        else:
            self.mid_channels = in_channels
            self.skip_channels = 0

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, self.mid_channels, 1, 1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, self.mid_channels, kernel_size=2, stride=2)

        self.conv = MultiConv(self.mid_channels + self.skip_channels, out_channels, conv_num=conv_num)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x2 is not None:
            assert self.skip == True
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

            x1 = torch.cat([x2, x1], dim=1)
        else:
            assert self.skip == False

        x = self.conv(x1)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )


class UNet_ConvFormer(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64,
                 size='base',
                 pretrained=False):
        super(UNet_ConvFormer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        if size == 'base':
            self.seg_backbone = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                                          num_heads=9, mlp_ratio=4, qkv_bias=True)

            self.up1 = Up(base_c * 24, base_c * 12, bilinear, conv_num=3)
            self.up2 = Up(base_c * 12, base_c * 6, bilinear, conv_num=3)
            self.up3 = Up(base_c * 6, base_c * 1, bilinear, conv_num=2)
            self.up4 = Up(base_c * 1, base_c, bilinear, conv_num=2, skip=False)
            self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x_out_4, x_out_3, x_out_2, x_out_1, x_t = self.seg_backbone(x)

        x = self.up1(x_out_4, x_out_3)
        x = self.up2(x, x_out_2)
        x = self.up3(x, x_out_1)
        x = self.up4(x, None)
        logits = self.out_conv(x)

        return logits, x_t


class BasicLinearBlock(nn.Module):
    def __init__(self, in_neuron, out_neuron):
        super(BasicLinearBlock, self).__init__()
        self.fc = nn.Linear(in_neuron, out_neuron)
        self.bn1 = nn.BatchNorm1d(out_neuron)
        self.activate = nn.ReLU()
    
    def forward(self, x):
        x_o = self.fc(x)
        x = self.bn1(x_o)
        x = self.activate(x)
        
        return x, x_o


class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes,
                 iter_num_dilate=25, iter_num_shrink=15):
        super(Classifier, self).__init__()
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.conv_more = BasicConv(embed_dim, embed_dim)
        self.trans_cls_head = nn.Linear(768, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mtr_attn = MultiRegionAttention(dim=embed_dim, num_heads=8, qkv_bias=True,
                                             iter_num_dilate=iter_num_dilate,
                                             iter_num_shrink=iter_num_shrink)
        self.linear_final = BasicLinearBlock(embed_dim * 4, 768)

    def forward(self, x_t, mask=None):
        B, N, C = x_t.shape
        H = W = int(math.sqrt(N - 1))
        x_t = x_t[:, 1:]

        x_t = x_t + self.mtr_attn(self.norm1(x_t), mask)

        x_t = self.trans_norm(x_t)
        x_r = x_t.transpose(1, 2).reshape(B, C, H, W)

        x_r = self.conv_more(x_r)
        x_r = torch.flatten(self.avgpool(x_r), 1)
        x_r, x_feat = self.linear_final(x_r)
        out_cls = self.trans_cls_head(x_r)

        return out_cls, x_feat


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def mask_dilate(mask, iter=25):
    dilate = mask
    for i in range(iter):
        dilate = soft_dilate(dilate)
    return dilate


def mask_shrink(mask, iter=5):
    dilate = 1 - mask
    for i in range(iter):
        dilate = soft_dilate(dilate)
    shrink = 1 - dilate
    return shrink


def cnn_mask_to_transformer_att(mask_1, mask_2, scale):
    mask_1 = F.interpolate(mask_1, size=None, scale_factor=scale, mode='bilinear', align_corners=None)
    mask_2 = F.interpolate(mask_2, size=None, scale_factor=scale, mode='bilinear', align_corners=None)
    mask_h = mask_2.flatten(2)
    mask_v = mask_1.flatten(2).transpose(-1, -2)
    mask = (mask_v @ mask_h).unsqueeze(1)
    mask = nn.Hardtanh()(mask + mask.transpose(-1, -2))
    return mask


def combine_att_masks(mask_intra, mask_peri, wbs, scale):
    att_mask_intra = cnn_mask_to_transformer_att(mask_intra, mask_intra, scale) * wbs[0][0] + wbs[0][1]
    att_mask_peri = cnn_mask_to_transformer_att(mask_peri, mask_peri, scale) * wbs[1][0] + wbs[1][1]
    att_mask_list = [att_mask_intra, att_mask_peri, att_mask_intra, att_mask_peri,
                     att_mask_intra, att_mask_peri, att_mask_intra, att_mask_peri]
    return torch.concat(att_mask_list, dim=1)


class MultiRegionAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 qk_scale=None,
                 iter_num_dilate=25,
                 iter_num_shrink=15):
        super(MultiRegionAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.iter_num_dilate = iter_num_dilate
        self.iter_num_shrink = iter_num_shrink
        self.wb_att = nn.Parameter(torch.Tensor([[1.0, 0.2], [1.0, 0.2]]), requires_grad=True)

    def forward(self, x, mask=None):
        B, N, C = x.shape

        if mask is not None:
            mask_intra = mask
            mask_dil = mask_dilate(mask, iter=self.iter_num_dilate)
            mask_shr = mask_shrink(mask, iter=self.iter_num_shrink)

            mask_peri = mask_dil - mask_shr
            scale = int(math.sqrt(N)) / 256
            assert scale == 1 / 16

            mask_trans_att = combine_att_masks(mask_intra, mask_peri, self.wb_att, scale)
            assert mask_trans_att.shape[1] == self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if mask is not None:
            attn = attn * mask_trans_att

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        return x


class ConvFormer_MTL(nn.Module):
    def __init__(self):
        super(ConvFormer_MTL, self).__init__()

        self.seg_branch = UNet_ConvFormer(pretrained=False, size='base')
        self.cls_branch = Classifier(embed_dim=576, num_classes=2)

    def forward(self, x, mask=None):

        logits, x_t = self.seg_branch(x)

        if mask is not None:
            out_cls, feat_cls = self.cls_branch(x_t, mask)
        else:
            out_cls, feat_cls = self.cls_branch(x_t, logits)
        
        if not self.training:
            out_cls = torch.softmax(out_cls, dim=1)
            # print("use softmax for out.")

        return logits, out_cls, feat_cls


