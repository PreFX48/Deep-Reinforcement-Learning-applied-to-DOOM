import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(height_width, kernel_size, stride):
    # Compute the output dimensions when applying a convolutional layer. 
    height, width = height_width
    new_height = (height - (kernel_size - 1) - 1) // stride + 1
    new_width = (width - (kernel_size - 1) - 1) // stride + 1
    return new_height, new_width


class OracleNetwork(nn.Module):
    def __init__(self, w=120, h=160, stack_size=4):
        super(OracleNetwork, self).__init__()

        # Conv Module
        self.conv_1 = nn.Conv2d(in_channels=stack_size, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=7, stride=1, padding=3)
        self.conv_4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        return x
