import torch
from torch import nn

import torch
from torch import nn

# Matching Output Size for eahc Cornet Layer: (224*224*3), (112*112*64), (56*56*128), (28*28*256), (14*14*512), (1*1*1000)
MLP_LAYER_SIZES = [112*112*64, 56*56*128, 28*28*256, 14*14*512]  # Subsequent layer sizes
IMAGE_DIM = (224, 224,3)  # Default Image dimensions ( H, W,C,)

class MLP(nn.Module):
    def __init__(self, input_size=IMAGE_DIM, layer_sizes=MLP_LAYER_SIZES, num_classes=1000):
        super().__init__()

        
        # Dynamically calculate the input layer size based on image dimensions
        if isinstance(input_size, tuple):
            input_features = input_size[0] * input_size[1] * input_size[2]  # H * W * C
        else:
            input_features = input_size  # Assuming input_size is already the number of input features
        
        # Update the first layer size to match the calculated input features
        updated_layer_sizes = [input_features] + layer_sizes
        
        modules = []
        for i in range(len(updated_layer_sizes)-1):
            modules.append(nn.Linear(updated_layer_sizes[i], updated_layer_sizes[i+1]))
            # Apply ReLU non-linearity between layers except the last layer
            if i < len(updated_layer_sizes) - 2:  # No ReLU before the final classification layer
                modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(updated_layer_sizes[-1], num_classes))
        self.layers = nn.Sequential(*modules)
    
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


'''
# ----------------------------------------------------- #
# Instantiate the MLP models with dynamic input size handling? 
# ----------------------------------------------------- #
# Example of defining image dimensions and layer sizes
# image_dim = (224, 224,3)  # Image dimensions (C, H, W)
# mlp_layer_sizes = [112*112*64, 56*56*128, 28*28*256, 14*14*512]  # Subsequent layer sizes
# MLP_R = MLP(image_dim)
# MLP_Z = MLP(image_dim, mlp_layer_sizes)
# MLP_RT = MLP(image_dim, mlp_layer_sizes)
# MLP_S = MLP(image_dim, mlp_layer_sizes)



# ----------------------------------------------------- #
# How many parameters in the model?
# ----------------------------------------------------- #
# Calculating the total parameters for the MLP model given the layer sizes and input size

# Input to first hidden layer
# input_to_first_layer = (150528 * (112*112*64)) + (112*112*64)

# # First to second hidden layer
# first_to_second = ((112*112*64) * (56*56*128)) + (56*56*128)

# # Second to third hidden layer
# second_to_third = ((56*56*128) * (28*28*256)) + (28*28*256)

# # Third to fourth hidden layer
# third_to_fourth = ((28*28*256) * (14*14*512)) + (14*14*512)

# # Fourth to output layer
# fourth_to_output = ((14*14*512) * 1000) + 1000

# # Total parameters
# total_parameters = input_to_first_layer + first_to_second + second_to_third + third_to_fourth + fourth_to_output

# total_parameters: 543,910,149,096, 544 billion parameters?


# ----------------------------------------------------- #
# How much memoery in the model?
# ----------------------------------------------------- #

To estimate the storage required for the MLP model with approximately \(543,910,149,096\) parameters, we need to consider the data type used for storing these parameters. In neural networks, parameters (weights and biases) are typically stored as 32-bit floating-point numbers (float32). Each float32 number requires 4 bytes of storage.

The formula to calculate the storage needed is:

\[ \text{Total Storage Needed} = \text{Total Parameters} \times \text{Bytes per Parameter} \]

Substituting the given values:

\[ \text{Total Storage Needed} = 543,910,149,096 \times 4 \, \text{bytes} \]

Let's compute the exact storage requirement.

The total storage required for the MLP model with approximately \(543,910,149,096\) parameters is around \(2026.22\) gigabytes (GB). 
This significant storage requirement highlights the impracticality of using such a large MLP for image processing tasks, especially in comparison to more parameter-efficient architectures like CNNs.

Which means 2TB(2000GB)of memory is required to store the model parameters!

'''


