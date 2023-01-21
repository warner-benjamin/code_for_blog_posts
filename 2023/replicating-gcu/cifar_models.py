import torch
import torch.nn as nn
from torch.autograd import Function


@torch.jit.script
def _gcu_jit_fwd(x):
    return x * torch.cos(x)

@torch.jit.script
def _gcu_jit_bwd(x, grad_output):
    return grad_output.mul(torch.cos(x) - x * torch.sin(x))

class _GCUJitAutoFn(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x)
        return _gcu_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        x = ctx.saved_tensors[0]
        return _gcu_jit_bwd(x, grad_output), None

class GCUJit(nn.Module):
    "TorchScript Growing Cosine Unit from https://arxiv.org/abs/2108.12943"
    def forward(self, x):
        return _GCUJitAutoFn.apply(x)


class GCU(nn.Module):
    "TorchScript Growing Cosine Unit from https://arxiv.org/abs/2108.12943"
    def forward(self, x):
        return x * torch.cos(x)


# _init_weights modified from PyTorch Image Models
# timm - Apache-2.0 license - Copyright (c) 2019 Ross Wightman

def _init_weights(module: nn.Module, act_cls=GCU):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        if act_cls==GCU:
            nn.init.kaiming_uniform_(module.weight)
        else:
            nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    for m in module.children():
        _init_weights(m, act_cls=act_cls)



class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, act_cls:nn.Module=nn.ReLU,
                 pool:bool=False, avg_pool:bool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act  = act_cls()
        if pool and avg_pool:
            self.pooling = nn.AvgPool2d(2)
        elif pool:
            self.pooling = nn.MaxPool2d(2)
        else:
            self.pooling = nn.Identity()
        self.pool = pool

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.pool:
            x = self.pooling(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, act_cls:nn.Module):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, act_cls)
        self.conv2 = ConvBlock(out_channels, out_channels, act_cls)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x

class CifarResNet(nn.Module):
    def __init__(self, num_classes:int, act_cls:nn.Module):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64, act_cls=act_cls),
            ConvBlock(64, 128, act_cls=act_cls, pool=True, avg_pool=False),
            ResBlock(128, 128, act_cls=act_cls),
            ConvBlock(128, 256, act_cls=act_cls, pool=True),
            ResBlock(256, 256, act_cls=act_cls),
            ConvBlock(256, 512, act_cls=act_cls, pool=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        self.init_weights(act_cls)

    @torch.jit.ignore
    def init_weights(self, act_cls):
        _init_weights(self, act_cls)

    def forward(self, x):
        return self.model(x)


class CifarVGG(nn.Module):
    def __init__(self, num_classes=10, act_cls=nn.ReLU, hct_cls=nn.ReLU, drop=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            act_cls(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            act_cls(),
            nn.MaxPool2d(2),
            nn.Dropout(drop),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            act_cls(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            act_cls(),
            nn.MaxPool2d(2),
            nn.Dropout(drop),
            nn.Flatten(),
            nn.Linear(2304, 512),
            hct_cls(),
            nn.Dropout(drop),
            nn.Linear(512, num_classes)
        )
        self.init_weights(act_cls)

    @torch.jit.ignore
    def init_weights(self, act_cls):
        _init_weights(self, act_cls)

    def forward(self, x):
        return self.model(x)



def cifarModels(cifar:int=10, model:str='vgg', act_cls:str='gcu', drop:float=0.5, jit=False):
    hct_cls=nn.ReLU
    if act_cls=='relu':
        act_cls=nn.ReLU
    elif act_cls=='gcu':
        act_cls=GCU if jit else GCUJit
        hct_cls=GCU if jit else GCUJit
    elif act_cls=='gcuh':
        act_cls=GCU if jit else GCUJit
    elif act_cls=='mish':
        act_cls=nn.Mish
        hct_cls=nn.Mish
    elif act_cls=='mishh':
        act_cls=nn.Mish
    elif act_cls=='silu':
        act_cls=nn.SiLU
        hct_cls=nn.SiLU
    elif act_cls=='siluh':
        act_cls=nn.SiLU
    elif act_cls=='gelu':
        act_cls=nn.GELU
        hct_cls=nn.GELU
    elif act_cls=='geluh':
        act_cls=nn.GELU
    else:
        raise ValueError(f'Unsupported {act_cls=}')

    if model=='vgg':
        return CifarVGG(num_classes=cifar, act_cls=act_cls, hct_cls=hct_cls, drop=drop)
    elif model=='resnet':
        return CifarResNet(num_classes=cifar, act_cls=act_cls)
    else:
        raise ValueError(f'Unsupported {model=}')