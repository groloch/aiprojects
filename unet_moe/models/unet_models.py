from torch.nn import Module, ModuleList
from torch.nn import Conv2d, ReLU, BatchNorm2d, MaxPool2d, UpsamplingNearest2d, AdaptiveAvgPool2d
from torch.nn import Linear
import torch

class _CountedModule(Module):
    def parameters_count(self):
        sum = 0
        for param in self.parameters():
            prod = 1
            for k in param.shape:
                prod *= k
            sum += prod
        return sum
    

class ConvolutionBlock(_CountedModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 do_bn=True, do_relu=True) -> None:
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, padding='same')
        
        self.do_bn = do_bn
        self.bn = BatchNorm2d(out_channels) if self.do_bn else None
        
        self.do_relu = do_relu
        self.relu = ReLU() if self.do_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.do_bn:
            x = self.bn(x)
        if self.do_relu:
            x = self.relu(x)
        return x
    
class LinearBlock(_CountedModule):
    def __init__(self, in_features, out_features, do_relu=True) -> None:
        super().__init__()

        self.linear = Linear(in_features, out_features)
        
        
        self.do_relu = do_relu
        self.relu = ReLU() if do_relu else None

    def forward(self, x):
        print(x.shape)
        x = self.linear(x)

        if self.do_relu:
            x = self.do_relu(x)

        return x
    

class UnetModel(_CountedModule):
    def __init__(self, in_channels, out_channels, f0, encoder_blocks=4) -> None:
        super().__init__()

        self.pre_encoder = ConvolutionBlock(in_channels, f0, (3, 3))

        self.encoders = ModuleList()
        for k in range(0, encoder_blocks):
            self.encoders.append(ConvolutionBlock(f0*2**k, f0*2**(k+1), (3, 3)))

        self.latent_conv = ConvolutionBlock(f0*2**encoder_blocks, f0*2**encoder_blocks, (3, 3))

        self.decoders = ModuleList()
        for k in range(encoder_blocks, 0, -1):
            self.decoders.append(ConvolutionBlock(f0*2**k, f0*2**(k-1), (3, 3)))

        self.head = ConvolutionBlock(f0, out_channels, (3, 3))

        self.maxpool = MaxPool2d((2, 2))
        self.upsampling = UpsamplingNearest2d(scale_factor=(2, 2))
        
    def forward(self, x):
        x = self.pre_encoder(x)

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.maxpool(x)

        x = self.latent_conv(x)

        for decoder in self.decoders:
            x = self.upsampling(x)
            skip = skips.pop(-1)
            x = torch.add(x, skip)
            x = decoder(x)

        x = self.head(x)
        return x


class Router(_CountedModule):
    def __init__(self, in_channels, n_experts, smooth=1) -> None:
        super().__init__()
        
        self.pooling = AdaptiveAvgPool2d(1)
        self.linear = LinearBlock(in_channels, n_experts, do_relu=False)

    def forward(self, x):
        x = self.pooling(x)
        x = self.linear(x[:, :, 0, 0])
        x = torch.softmax(x, dim=-1)

        return x


class UnetMOEModel(_CountedModule):
    def __init__(self, in_channels, out_channels, f0, encoder_blocks=4, n_experts=4) -> None:
        super().__init__()

        self.pre_encoder = ConvolutionBlock(in_channels, f0, (3, 3))
        
        
        self.pre_encoder = ConvolutionBlock(in_channels, f0, (3, 3))

        self.encoders = ModuleList()
        for k in range(0, encoder_blocks):
            self.encoders.append(ConvolutionBlock(f0*2**k, f0*2**(k+1), (3, 3)))

        f_lat = f0*2**encoder_blocks
        self.router = Router(f_lat, n_experts)
        self.experts = ModuleList()
        
        for k in range(n_experts):
            self.experts.append(ConvolutionBlock(f_lat, f_lat, (2, 2)))

        self.decoders = ModuleList()
        for k in range(encoder_blocks, 0, -1):
            self.decoders.append(ConvolutionBlock(f0*2**k, f0*2**(k-1), (3, 3)))

        self.head = ConvolutionBlock(f0, out_channels, (3, 3))

        self.maxpool = MaxPool2d((2, 2))
        self.upsampling = UpsamplingNearest2d(scale_factor=(2, 2))

    def forward(self, x):
        x = self.pre_encoder(x)

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.maxpool(x)
        
        routing = self.router(x)

        experts_out = []
        for k, expert in enumerate(self.experts):
            x = expert(x)
            experts_out.append(x)

        x = torch.stack(experts_out, dim=1) * routing.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = torch.sum(x, dim=1)

        for decoder in self.decoders:
            x = self.upsampling(x)
            x = torch.add(x, skips.pop(-1))
            x = decoder(x)

        x = self.head(x)

        return x
    