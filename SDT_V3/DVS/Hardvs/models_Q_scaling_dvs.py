# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
# from spikingjelly.clock_driven.neuron import (
#     MultiStepParametricLIFNode,
#     MultiStepLIFNode,
# )
from Qtrick_architecture.clock_driven import neuron
from Qtrick_architecture.clock_driven import surrogate
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial


# class Quant(torch.autograd.Function):
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def forward(ctx, i, min_value=0, max_value=8):
#         ctx.min = min_value
#         ctx.max = max_value
#         ctx.save_for_backward(i)
#         return torch.round(torch.clamp(i, min=min_value, max=max_value))

#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         i, = ctx.saved_tensors
#         grad_input[i < ctx.min] = 0
#         grad_input[i > ctx.max] = 0
#         return grad_input, None, None


# class MultiSpike_norm(nn.Module):
#     def __init__(
#         self,
#         Norm = 8,
#         ):
#         super().__init__()
#         self.spike = Quant()
#         self.Norm = Norm
#     def forward(self, x):
#         return self.spike.apply(x) / (self.Norm)
#     def __repr__(self):
#         return f"MultiSpike_norm(Norm={self.Norm})"



#===========================

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x

class SepRepConv(nn.Module): #放在Sepconv最后一个1*1卷积，采用3*3分组+1*1降维的方式实现，能提0.5个点。之后可以试试改成1*1降维和3*3分组
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
#         conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, 1,0, groups=in_channel, bias=False),  #这里也是分组卷积
            nn.Conv2d(in_channel, out_channel, 1,  1,0, groups=group, bias=False),
        )

        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
#             MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch'), #1111
            nn.Conv2d(in_channel, in_channel, kernel_size, 1,0, groups=in_channel, bias=False),  #这里也是分组卷积
#             MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch'), #1111
            nn.Conv2d(in_channel, out_channel, 1,  1,0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,  #7,3
                 padding=1):
        super().__init__()
        padding = int((kernel_size -1)/2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size, #7*7
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
#         self.pwconv3 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias,groups=1)
        self.pwconv3=SepRepConv(med_channels, dim)  #这里将sepconv最后一个卷积替换为重参数化卷积  大概提0.5个点，可以保留

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)


        self.lif1 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        self.lif2 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        self.lif3 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

    def forward(self, x):
        T, B, C, H, W = x.shape #输入256
#         print("x.shape:",x.shape)
        x = self.lif1(x) #x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)  # flatten：从第0维开始，展开到第一维
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x

class MS_AllConvBlock_yolo(nn.Module):  #在这个模式中，将repconv两部分全部替换为普通卷积
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)  #内部扩张2倍
        self.mlp_ratio = mlp_ratio
        if group == True:
            self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio),3,g=4)  #136版本
        else:
            self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio),3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim,3)



    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x

# class MS_DownSampling_yolo(nn.Module):
#     def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
#         super().__init__()
#
#         self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
#
#         self.encode_bn = nn.BatchNorm2d(embed_dims)
#         if not first_layer:
#             self.encode_lif = MultiStepLIFNode(detach_reset=True,v_reset=None, backend='torch')
#         # self.pool = nn.MaxPool2d(kernel_size=2)
#
#     def forward(self, x):
#         T, B, _, _, _ = x.shape
#
#
#         if hasattr(self, "encode_lif"): #如果不是第一层
#             # x_pool = self.pool(x)
#             x = self.encode_lif(x)
#
#
#         x = self.encode_conv(x.flatten(0, 1))
#         _, C, H, W = x.shape
#         x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
#
#         return x

class MS_DownSampling_yolo(nn.Module):
    def __init__(
            self,
            in_channels=2,
            embed_dims=256,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,
            D_Norm=8

    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.D_Norm = D_Norm
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

    def forward(self, x):
        T, B, _, _, _ = x.shape
        # x = x.to(torch.float16)
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x) / self.D_Norm
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W)

        return x

class MS_ConvBlock_yolo(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, full=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.full = full
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)  # 内部扩张2倍
        self.mlp_ratio = mlp_ratio

        self.lif1 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        self.lif2 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio))  # 137以外的模型，在第一个block不做分组
        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进


    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        #         x = self.halfconv(self.lif1(x).flatten(0, 1)).reshape(T, B, C, H, W) #完成repconv的前半部分
        #         x = self.standardconv(x) #升维
        #         print("==============self.lif1(x)",self.lif1(x))

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
        # repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x

class Efficient_Spikeyolo_scaling(nn.Module):
    def __init__(
            self,
            in_channels=2,
            num_classes=11,
            embed_dim=[64, 128, 256,512],
            mlp_ratios=[4, 4, 4],
            drop_path_rate=0.0,
            depths=3,

    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # embed_dim = [64, 128, 256, 512]

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule


        self.downsample1 = MS_DownSampling_yolo(
            in_channels=in_channels,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,


        )

        self.ConvBlock1 = nn.ModuleList(
            [
                MS_AllConvBlock_yolo(input_dim=embed_dim[0], mlp_ratio=mlp_ratios[0])
                for j in range(int(depths * 0.34))
            ]
        )

        self.downsample2 = MS_DownSampling_yolo(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.ConvBlock2 = nn.ModuleList(
            [
                MS_AllConvBlock_yolo(input_dim=embed_dim[1], mlp_ratio=mlp_ratios[1])
                for j in range(int(depths * 0.67))
            ]
        )


        self.downsample3 = MS_DownSampling_yolo(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.block3 = nn.ModuleList(
            [
                MS_ConvBlock_yolo(input_dim=embed_dim[2], mlp_ratio=mlp_ratios[2])
                for j in range(int(depths * 1))
            ]
        )

        self.downsample4 = MS_DownSampling_yolo(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.block4 = nn.ModuleList(
            [
                MS_ConvBlock_yolo(input_dim=embed_dim[3], mlp_ratio=mlp_ratios[3], )
                for j in range(int(depths * 0.34))
            ]
        )

        self.head = (
            nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        )
        self.spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample1(x)
        for blk in self.ConvBlock1:
            x = blk(x)


        x = self.downsample2(x)
        for blk in self.ConvBlock2:
            x = blk(x)


        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)

        return x  # T,B,C,N

    def forward(self, x):
        # x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1) # dvs输入本身就带有时间维度
        x = x.transpose(0, 1).contiguous()
        x = self.forward_features(x)  # T,B,C,H,W
        x = x.flatten(3).mean(3)
        x = self.spike(x)
        x = self.head(x).mean(0)
        return x




def Efficient_Spikeyolo_scaling_8_19M(**kwargs):
    model = Efficient_Spikeyolo_scaling(
        in_channels=2,
        num_classes=300,
        embed_dim=[64, 128, 256, 360],
        mlp_ratios=[4,4,3,1.5],
        depths=3,
    )
    return model

#===========================












class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
        D_Norm=8,
        
    ):
        super().__init__()
        self.D_Norm = D_Norm
        med_channels = int(expansion_ratio * dim)
        self.spike1 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
            )
        self.spike2 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):

        T, B, C, H, W = x.shape
        
        x = self.spike1(x) / self.D_Norm
            
        x = self.pwconv1(x.flatten(0,1)).reshape(T, B, -1, H, W)
        
        x = self.spike2(x) / self.D_Norm
            
        x = self.dwconv(x.flatten(0,1)).reshape(T, B, -1, H, W)

        x = self.spike3(x) / self.D_Norm

        x = self.pwconv2(x.flatten(0,1)).reshape(T, B, C, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        D_Norm=8,
        
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim, D_Norm=D_Norm)

        self.mlp_ratio = mlp_ratio
        self.D_Norm = D_Norm

        self.spike1 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x) / self.D_Norm
        x = self.bn1(self.conv1(x.flatten(0,1))).reshape(T, B, self.mlp_ratio * C, H, W)
        x = self.spike2(x) / self.D_Norm
        x = self.bn2(self.conv2(x.flatten(0,1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0, D_Norm=8,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.D_Norm = D_Norm
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_spike(x) / self.D_Norm
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x) / self.D_Norm
        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class MS_Attention_linear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        D_Norm = 8,
        lamda_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.D_Norm = D_Norm
        self.num_heads = num_heads
        self.scale = (dim//num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim*lamda_ratio), 1, 1, bias=False), nn.BatchNorm2d(int(dim*lamda_ratio)))
        
        self.v_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

        self.attn_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())


        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim*lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )


    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        C_v = int(C*self.lamda_ratio)

        x = self.head_spike(x) / self.D_Norm

        q = self.q_conv(x.flatten(0,1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0,1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0,1)).reshape(T, B, C_v, H, W)

        q = self.q_spike(q) / self.D_Norm #11111111
        # q = self.q_spike(q) / 4
        q = q.flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
      
        k = self.k_spike(k) / self.D_Norm
        # k = self.k_spike(k) / 4
        k = k.flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_spike(v) / self.D_Norm
        # v = self.v_spike(v) / 4 #1111111
        v = v.flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C_v // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = q @ k.transpose(-2, -1)
        x = (x @ v) * (self.scale*2)

        x = x.transpose(3, 4).reshape(T, B, C_v, N).contiguous()
        x = self.attn_spike(x) / self.D_Norm
        x = x.reshape(T, B, C_v, H, W)
        x = self.proj_conv(x.flatten(0,1)).reshape(T, B, C, H, W)

        return x


class MS_Block_Spike_SepConv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        init_values = 1e-6,
        D_Norm=8,
    ):
        super().__init__()

        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1, D_Norm=D_Norm)

        self.attn = MS_Attention_linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4,
            D_Norm=D_Norm
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, D_Norm=D_Norm)

        self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        
    def forward(self, x):
        x = x + self.conv(x) * self.layer_scale1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x + self.attn(x) * self.layer_scale2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x + self.mlp(x) * self.layer_scale3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return x


class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        init_values=1e-6,
        T=None,
        D_Norm = 8,
    ):
        super().__init__()

        self.attn = MS_Attention_linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4,
            D_Norm=D_Norm
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, D_Norm=D_Norm)

        # self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        # self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # x = x + self.attn(x) * self.layer_scale1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # 是否使用layerscale待定
        # x = x + self.mlp(x) * self.layer_scale2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        D_Norm=8
        
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.D_Norm = D_Norm
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x) / self.D_Norm
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W)

        return x


class Efficient_SpikeFormer_scaling(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dim=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        D_Norm=4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.D_Norm = D_Norm
        # embed_dim = [64, 128, 256, 512]

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            D_Norm=self.D_Norm,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios, D_Norm=self.D_Norm)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            D_Norm=self.D_Norm,
            
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios, D_Norm=self.D_Norm)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            D_Norm=self.D_Norm,
            
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios, D_Norm=self.D_Norm)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios, D_Norm=self.D_Norm,)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            D_Norm=self.D_Norm,
            
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    D_Norm=self.D_Norm,
                    
                )
                for j in range(int(depths*0.75))
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            D_Norm=self.D_Norm,
            
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    D_Norm=self.D_Norm,
                    
                )
                for j in range(int(depths*0.25))
            ]
        )
        
        self.head = (
            nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        )
        self.spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)

        return x  # T,B,C,N

    def forward(self, x):
        # x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1) # dvs输入本身就带有时间维度
        x = x.transpose(0, 1).contiguous()
        x = self.forward_features(x) # T,B,C,H,W
        x = x.flatten(3).mean(3)
        x = self.spike(x)
        x = self.head(x).mean(0)
        return x


def Efficient_Spiking_Transformer_scaling_2_8M_gesture(**kwargs):
    model = Efficient_SpikeFormer_scaling(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        num_classes=11,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=2,
        sr_ratios=1,
        D_Norm=8, 
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_scaling_8_19M(**kwargs):
    model = Efficient_SpikeFormer_scaling(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        num_classes=300,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        D_Norm=1, #111111
        **kwargs,
    )
    return model


def Efficient_Spiking_Transformer_scaling_8_30M(**kwargs):
    model = Efficient_SpikeFormer_scaling(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[96, 192, 384, 480],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_scaling_8_50M(**kwargs):
    model = Efficient_SpikeFormer_scaling(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[120, 240, 480, 600],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model

import time
if __name__ == "__main__":
    model = Efficient_Spiking_Transformer_scaling_8_19M()
    print(model)
    x = torch.randn(8,1,2,128,128)
    y = model(x)
    torchinfo.summary(model, (8, 1, 2, 128, 128), device='cpu')
    print(y.shape)