from collections import OrderedDict
import torch
from torch import nn

""" Architecture of CORnet Model
V1, V2, V4, IT, and decoder are the names of the layers in the CORnet model.
Output Size for eahc Cornet Layer: (224*224*3), (112*112*64), (56*56*128), (28*28*256), (14*14*512), (1*1*1000)

To calculate the output feature map size for each layer in the CORblock_Z architecture, we need to apply the formulas for convolutional and pooling layers. The general formula for calculating the output size of a layer is given by:

\[ \text{Output size} = \left( \frac{\text{Input size} - \text{Kernel size} + 2 \times \text{Padding}}{\text{Stride}} \right) + 1 \]

For max pooling layers, the formula is similar but typically uses a stride equal to the kernel size (unless otherwise specified), which simplifies the calculation.

Given the CORblock_Z architecture specified, let's calculate the output sizes layer by layer assuming an input image size of \(224 \times 224\) pixels (a common size for CNN inputs). Note that each CORblock_Z instance includes a convolution followed by a ReLU and then a max pooling operation.

### V1 Layer:
- **Input size:** \(224 \times 224\)
- **Convolution:** Kernel size = 7, Stride = 2, Padding = \(7 // 2 = 3.5\) (rounding to 3 for simplicity)
  - \( \text{Output size} = \left( \frac{224 - 7 + 2 \times 3}{2} \right) + 1 = 112.5 \) (rounding down to 112 for simplicity)
- **Max Pooling:** Kernel size = 3, Stride = 2, Padding = 1
  - \( \text{Output size} = \left( \frac{112 - 3 + 2 \times 1}{2} \right) + 1 = 56 \)

### V2 Layer:
- **Input size:** \(56 \times 56\)
- **Convolution:** Kernel size = 3, Stride = 1, Padding = \(3 // 2 = 1\)
  - \( \text{Output size} = \left( \frac{56 - 3 + 2 \times 1}{1} \right) + 1 = 56 \) (size remains the same due to stride 1 and padding)
- **Max Pooling:** Kernel size = 3, Stride = 2, Padding = 1
  - \( \text{Output size} = \left( \frac{56 - 3 + 2 \times 1}{2} \right) + 1 = 28 \)

### V4 Layer:
- **Input size:** \(28 \times 28\)
- **Convolution:** Kernel size = 3, Stride = 1, Padding = 1
  - \( \text{Output size} = \left( \frac{28 - 3 + 2 \times 1}{1} \right) + 1 = 28 \)
- **Max Pooling:** Kernel size = 3, Stride = 2, Padding = 1
  - \( \text{Output size} = \left( \frac{28 - 3 + 2 \times 1}{2} \right) + 1 = 14 \)

### IT Layer:
- **Input size:** \(14 \times 14\)
- **Convolution:** Kernel size = 3, Stride = 1, Padding = 1
  - \( \text{Output size} = \left( \frac{14 - 3 + 2 \times 1}{1} \right) + 1 = 14 \)
- **Max Pooling:** Kernel size = 3, Stride = 2, Padding = 1
  - \( \text{Output size} = \left( \frac{14 - 3 + 2 \times 1}{2} \right) + 1 = 7 \)

### Decoder:
- After the IT layer, an Adaptive Average Pooling layer reduces the size to \(1 \times 1\) before flattening and passing through a fully connected layer.

This breakdown gives you the feature map sizes at the output of each layer in the CORnet_Z model, given an initial input image size of \(224 \times 224\).
"""

'''Parameterts and Storage needed
To compare the CORnet model's size in terms of parameters and storage to the previously discussed MLP model, we need to calculate the number of parameters for each component of the CORnet model and then estimate its storage requirements.

### CORnet Model Parameter Calculation

1. **Convolutional Layers (Conv2d):** The number of parameters for a convolutional layer is given by \((\text{kernel width} \times \text{kernel height} \times \text{input channels} \times \text{output channels}) + \text{output channels}\) for the bias term.

2. **Linear Layer (Linear):** The number of parameters is \((\text{input features} \times \text{output features}) + \text{output features}\) for the bias.

Given the structure of CORnet_Z:

- **V1:** Conv2d(3, 64, kernel_size=7, stride=2) + MaxPool
  - Parameters: \((7 \times 7 \times 3 \times 64) + 64\)
- **V2:** Conv2d(64, 128) + MaxPool
  - Parameters: \((3 \times 3 \times 64 \times 128) + 128\)
- **V4:** Conv2d(128, 256) + MaxPool
  - Parameters: \((3 \times 3 \times 128 \times 256) + 256\)
- **IT:** Conv2d(256, 512) + MaxPool
  - Parameters: \((3 \times 3 \times 256 \times 512) + 512\)
- **Decoder:** Linear(512, 1000)
  - Parameters: \((512 \times 1000) + 1000\)

Let's compute the total parameters for the CORnet model.

The CORnet model has approximately \(2,071,656\) parameters.

### Storage Requirement for CORnet

To calculate the storage requirement, assuming each parameter is stored as a 32-bit floating-point number (4 bytes):

\[ \text{Storage Required (bytes)} = \text{Total Parameters} \times 4 \]

\[ \text{Storage Required (GB)} = \frac{\text{Storage Required (bytes)}}{1024^3} \]

Let's calculate the storage requirement for the CORnet model based on its total number of parameters.

The CORnet model requires approximately \(0.0077\) gigabytes (GB) of storage.

### Comparison with the MLP Model:

- **MLP Model:** Approximately \(543,910,149,096\) parameters, requiring about \(2026.22\) GB of storage.
- **CORnet Model:** Approximately \(2,071,656\) parameters, requiring about \(0.0077\) GB (\(7.72\) MB) of storage.

This comparison highlights the significant difference in efficiency between convolutional neural networks (CNNs) like CORnet and fully connected networks (MLPs). 
CNNs, by using shared weights and spatial hierarchies, drastically reduce the number of parameters and storage requirements, 
making them much more suitable for image processing tasks.

'''

HASH = '5930a990'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_R(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp=None, state=None, batch_size=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        x = self.norm1(x)
        x = self.nonlin1(x)

        state = self.output(x)
        output = state
        return output, state


class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


class CORblock_RT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp=None, state=None, batch_size=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        x = self.norm1(x)
        x = self.nonlin1(x)

        state = self.output(x)
        output = state
        return output, state


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output



class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x

def CORnet_Z():
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model

class CORnet_R(nn.Module):

    def __init__(self, times=5):
        super().__init__()
        self.times = times

        self.V1 = CORblock_R(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = CORblock_R(64, 128, stride=2, out_shape=28)
        self.V4 = CORblock_R(128, 256, stride=2, out_shape=14)
        self.IT = CORblock_R(256, 512, stride=2, out_shape=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000))
        ]))

    def forward(self, inp):
        outputs = {'inp': inp}
        states = {}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']

        for block in blocks[1:]:
            if block == 'V1':  # at t=0 input to V1 is the image
                inp = outputs['inp']
            else:  # at t=0 there is no input yet to V2 and up
                inp = None
            new_output, new_state = getattr(self, block)(inp, batch_size=outputs['inp'].shape[0])
            outputs[block] = new_output
            states[block] = new_state

        for t in range(1, self.times):
            for block in blocks[1:]:
                prev_block = blocks[blocks.index(block) - 1]
                prev_output = outputs[prev_block]
                prev_state = states[block]
                new_output, new_state = getattr(self, block)(prev_output, prev_state)
                outputs[block] = new_output
                states[block] = new_state

        out = self.decoder(outputs['IT'])
        return out


class CORnet_RT(nn.Module):

    def __init__(self, times=5):
        super().__init__()
        self.times = times

        self.V1 = CORblock_RT(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = CORblock_RT(64, 128, stride=2, out_shape=28)
        self.V4 = CORblock_RT(128, 256, stride=2, out_shape=14)
        self.IT = CORblock_RT(256, 512, stride=2, out_shape=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000))
        ]))

    def forward(self, inp):
        outputs = {'inp': inp}
        states = {}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']

        for block in blocks[1:]:
            if block == 'V1':  # at t=0 input to V1 is the image
                this_inp = outputs['inp']
            else:  # at t=0 there is no input yet to V2 and up
                this_inp = None
            new_output, new_state = getattr(self, block)(this_inp, batch_size=len(outputs['inp']))
            outputs[block] = new_output
            states[block] = new_state

        for t in range(1, self.times):
            new_outputs = {'inp': inp}
            for block in blocks[1:]:
                prev_block = blocks[blocks.index(block) - 1]
                prev_output = outputs[prev_block]
                prev_state = states[block]
                new_output, new_state = getattr(self, block)(prev_output, prev_state)
                new_outputs[block] = new_output
                states[block] = new_state
            outputs = new_outputs

        out = self.decoder(outputs['IT'])
        return out






def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot 
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model
