from torch.nn import Module
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d
from torch.nn import Linear

class ConvolutionBlock(Module):
    def __init__(self, in_channels, out_channels, fshape, padding='valid', 
                 do_bn=True, do_relu=True) -> None:
        super().__init__()

        self.conv = Conv2d(in_channels, out_channels, fshape, padding=padding)
        
        self.do_bn = do_bn
        self.bn = BatchNorm2d(out_channels) if self.do_bn else NotImplemented

        self.do_relu = do_relu
        self.relu = ReLU() if self.do_relu else None

    def forward(self, x):
        x = self.conv(x)
        
        if self.do_bn:
            x = self.bn(x)
        
        if self.do_relu:
            x = self.relu(x)
        return x


class SmallResnetModel(Module):
    def __init__(self, in_channels, out_classes) -> None:
        super().__init__()

        self.conv1 = ConvolutionBlock(in_channels, 4, (3, 3))

        