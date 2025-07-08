from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_


def trunc_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        trunc_normal_(m.weight, std=0.02)
    elif isinstance(m, nn.Parameter):
        trunc_normal_(m, std=.02)


class ResidualLayer(nn.Module):
    def __init__(self, rezero=False, layerscale=False, alpha=0.1, dim=1):
        super().__init__()
        self.rezero = rezero
        self.layerscale = layerscale
        self.layerscale_init = alpha

        if rezero:
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        elif layerscale:
            self.alpha = nn.Parameter(torch.ones(1, dim) * alpha, requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)

        self.dim = self.alpha.size(-1)

    def forward(self, x, x_res):
        if not self.rezero and not self.layerscale:
            return x + x_res
        return x * self.alpha + x_res


class AffineTransformLayer(nn.Module):
    def __init__(self, dim, decay_factor=1.):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim) * decay_factor, requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.init_decay = decay_factor

    def forward(self, x):
        return x * self.gamma + self.beta


class CKGConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34):
        super(CKGConv3d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K

        self.pe_dim = int(in_planes * ratio) + 1
        self.ffn_ratio = 1.0
        self.num_blocks = 1

        # Define channel dimensions explicitly
        hid_dim = self.in_planes  # Start with input channels
        ffn_dim = int(hid_dim * self.ffn_ratio)

        blocks = []
        # Attention blocks with proper channel tracking
        for _ in range(self.num_blocks):
            blocks.extend([
                nn.BatchNorm3d(hid_dim),
                nn.GELU(),
                nn.Conv3d(hid_dim, ffn_dim, 1, bias=True),
                nn.BatchNorm3d(ffn_dim),
                nn.GELU(),
                nn.Conv3d(ffn_dim, hid_dim, 1, bias=True),
                ResidualLayer(rezero=False, layerscale=False, dim=hid_dim)
            ])

        # Final transformation to K
        blocks.extend([
            nn.BatchNorm3d(hid_dim),
            nn.Conv3d(hid_dim, self.K, 1, bias=True)
        ])

        self.keys = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size, kernel_size),
            requires_grad=True
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None

        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

        self.apply(trunc_init_)

    def update_temperature(self):
        if self.temperature.item() > 1:
            self.temperature.data -= 3
            print('Change temperature to:', self.temperature.item())

    def propagate_attention(self, x):
        attn = self.avgpool(x)
        attn_res = attn  # Store for residual

        for i, layer in enumerate(self.keys):
            if isinstance(layer, ResidualLayer):
                attn = layer(attn, attn_res)
            else:
                attn = layer(attn)

        attn = attn.view(attn.size(0), -1)
        softmax_attention = F.softmax(attn / self.temperature, dim=1)
        return softmax_attention

    def forward(self, x):
        batch_size, in_planes, depth, height, width = x.size()

        softmax_attention = self.propagate_attention(x)

        x = x.view(1, -1, depth, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(
            batch_size * self.out_planes,
            self.in_planes // self.groups,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size
        )

        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(
                x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size
            )
        else:
            output = F.conv3d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size
            )

        output = output.view(batch_size, self.out_planes,
                             output.size(-3), output.size(-2), output.size(-1))
        return output


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=False, groups=groups))


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck, gate_factor, squeeze_rate, group_3x3, heads):
        super(_DenseLayer, self).__init__()
        self.conv_1 = CKGConv3d(
            in_channels,
            bottleneck * growth_rate,
            kernel_size=1,
            ratio=0.25,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            K=4,
            temperature=34
        )
        self.conv_2 = Conv(
            bottleneck * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            groups=group_3x3
        )

    def forward(self, x):
        x_ = x
        x = self.conv_1(x_)
        x = self.conv_2(x)
        x = torch.cat([x_, x], 1)
        return x


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck, gate_factor, squeeze_rate, group_3x3, heads):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck, gate_factor, squeeze_rate,
                                group_3x3, heads)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CGCdenseNet(nn.Module):
    def __init__(self, band, num_classes):
        super(CGCdenseNet, self).__init__()
        self.name = 'CGCdensenet'
        self.stages = [4, 6, 8]
        self.growth = [8, 16, 32]
        self.progress = 0.0
        self.init_stride = 2
        self.pool_size = 7
        self.bottleneck = 4
        self.gate_factor = 0.25
        self.squeeze_rate = 16
        self.group_3x3 = 4
        self.heads = 4

        self.features = nn.Sequential()
        self.num_features = 2 * self.growth[0]
        self.features.add_module('init_conv', nn.Conv3d(1, self.num_features, kernel_size=3, stride=self.init_stride,
                                                        padding=1, bias=False))
        for i in range(len(self.stages)):
            self.add_block(i)

        self.bn_last = nn.BatchNorm3d(self.num_features)
        self.relu_last = nn.ReLU(inplace=True)
        self.pool_last = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(self.num_features, num_classes)
        self.classifier.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def add_block(self, i):
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            bottleneck=self.bottleneck,
            gate_factor=self.gate_factor,
            squeeze_rate=self.squeeze_rate,
            group_3x3=self.group_3x3,
            heads=self.heads
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features)
            self.features.add_module('transition_%d' % (i + 1), trans)

    def forward(self, x, progress=None, threshold=None):
        features = self.features(x)
        features = self.bn_last(features)
        features = self.relu_last(features)
        features = self.pool_last(features)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    net = DydenseNet(200, 12)

    from torchsummary import summary

    summary(net, input_size=[(1, 200, 7, 7)], batch_size=1)

    from thop import profile

    input = torch.randn(1, 1, 200, 7, 7)
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))