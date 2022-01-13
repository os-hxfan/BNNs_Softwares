import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import torch
import torch.autograd.profiler as profiler

from software.models.dropout import BernoulliDropout
from software.utils import Flatten


class VGG(nn.Module):
    def __init__(
        self, input_size, output_size, args, dropout=[False, False, False, False, False]
    ):
        super(VGG, self).__init__()
        self.args = args
        self.layers = self.make_layers(
            [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            dropout=dropout,
        )
        self.layers.append(Flatten())
        self.layers.append(nn.Linear(512, 512, bias=False))
        self.layers.append(nn.ReLU(True))
        if dropout >= 0.2:
            self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(nn.Linear(512, 512, bias=False))
        self.layers.append(nn.ReLU(True))

        self.q = args.q
        if self.q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x, samples=1, profile=False):
        if not profile:
            return self._forward_no_profile(x)
        else:
            return self._forward_profile(x, samples)

    def _forward_no_profile(self, x):
        if self.q:
            x = self.quant(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if self.q:
            x = self.dequant(x)
        return F.softmax(x, dim=-1)

    def _forward_profile(self, x, samples):
        static = -1
        cache = None
        with profiler.record_function("static_part"):
            if self.q:
                x = self.quant(x)

            for i, layer in enumerate(self.layers):
                if isinstance(layer, BernoulliDropout):
                    static = i
                    cache = x
                    break
                x = layer(x)
            if static == -1:
                if self.q:
                    x = self.dequant(x)
                return F.softmax(x, dim=-1)
        out = []
        with profiler.record_function("dynamic_part"):
            for sample in range(samples):
                x = cache
                for i in range(static, len(self.layers)):
                    x = self.layers[i](x)

                if self.q:
                    x = self.dequant(x)
                x = F.softmax(x, dim=-1)
                out.append(x.detach())
            # out = torch.stack(out, dim=1).mean(dim=1)
        return out

    def make_layers(self, cfg, dropout=0):
        layers = []
        in_channels = 3
        conv_counter = 0
        for i, v in enumerate(cfg):
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if i == len(cfg) - 1 and dropout > 0:
                    layers += [BernoulliDropout(self.args.p)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if 1 - (conv_counter / 10) <= dropout and i > 0:
                    layers += [
                        BernoulliDropout(self.args.p),
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                conv_counter += 1
                in_channels = v
        return nn.ModuleList(layers)


class VGG_P(VGG):
    def __init__(self, input_size, output_size, args):
        super(VGG_P, self).__init__(input_size, output_size, args, dropout=0)
        self.layers.append(nn.Linear(512, output_size, bias=False))

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.layers,
            [
                ["0", "1", "2"],
                ["4", "5", "6"],
                ["8", "9", "10"],
                ["11", "12", "13"],
                ["15", "16", "17"],
                ["18", "19", "20"],
                ["22", "23", "24"],
                ["25", "26", "27"],
                ["30", "31"],
                ["32", "33"],
            ],
            inplace=True,
        )


class VGG_LL(VGG):
    def __init__(self, input_size, output_size, args):
        super(VGG_LL, self).__init__(input_size, output_size, args, dropout=0)
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(nn.Linear(512, output_size, bias=False))

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.layers,
            [
                ["0", "1", "2"],
                ["4", "5", "6"],
                ["8", "9", "10"],
                ["11", "12", "13"],
                ["15", "16", "17"],
                ["18", "19", "20"],
                ["22", "23", "24"],
                ["25", "26", "27"],
                ["30", "31"],
                ["32", "33"],
            ],
            inplace=True,
        )


class VGG_ONE_THIRD(VGG):
    def __init__(self, input_size, output_size, args):
        super(VGG_ONE_THIRD, self).__init__(input_size, output_size, args, dropout=0.2)
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(nn.Linear(512, output_size, bias=False))

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.layers,
            [
                ["0", "1", "2"],
                ["4", "5", "6"],
                ["8", "9", "10"],
                ["11", "12", "13"],
                ["15", "16", "17"],
                ["18", "19", "20"],
                ["22", "23", "24"],
                ["25", "26", "27"],
                ["31", "32"],
                ["34", "35"],
            ],
            inplace=True,
        )


class VGG_HALF(VGG):
    def __init__(self, input_size, output_size, args):
        super(VGG_HALF, self).__init__(input_size, output_size, args, dropout=0.4)
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(nn.Linear(512, output_size, bias=False))

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.layers,
            [
                ["0", "1", "2"],
                ["4", "5", "6"],
                ["8", "9", "10"],
                ["11", "12", "13"],
                ["15", "16", "17"],
                ["18", "19", "20"],
                ["23", "24", "25"],
                ["27", "28", "29"],
                ["33", "34"],
                ["36", "37"],
            ],
            inplace=True,
        )


class VGG_TWO_THIRD(VGG):
    def __init__(self, input_size, output_size, args):
        super(VGG_TWO_THIRD, self).__init__(
            input_size, output_size, args, dropout=2 / 3
        )
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(nn.Linear(512, output_size, bias=False))

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.layers,
            [
                ["0", "1", "2"],
                ["4", "5", "6"],
                ["8", "9", "10"],
                ["11", "12", "13"],
                ["16", "17", "18"],
                ["20", "21", "22"],
                ["25", "26", "27"],
                ["29", "30", "31"],
                ["35", "36"],
                ["38", "39"],
            ],
            inplace=True,
        )


class VGG_ALL(VGG):
    def __init__(self, input_size, output_size, args):
        super(VGG_ALL, self).__init__(input_size, output_size, args, dropout=1)
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(nn.Linear(512, output_size, bias=False))

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.layers,
            [
                ["0", "1", "2"],
                ["5", "6", "7"],
                ["10", "11", "12"],
                ["14", "15", "16"],
                ["19", "20", "21"],
                ["23", "24", "25"],
                ["28", "29", "30"],
                ["32", "33", "34"],
                ["38", "39"],
                ["41", "42"],
            ],
            inplace=True,
        )
