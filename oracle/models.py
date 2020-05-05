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
        self.conv_1 = nn.Conv2d(in_channels=stack_size, out_channels=16, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        self.drop2 = nn.Dropout2d(0.2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        self.drop3 = nn.Dropout2d(0.2)
        self.conv_4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        self.drop4 = nn.Dropout2d(0.2)
        self.conv_left = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        self.conv_shoot = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        self.conv_right = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

        self.action_to_layer = {
            (1, 0, 0): self.conv_left,
            (0, 1, 0): self.conv_shoot,
            (0, 0, 1): self.conv_right,
        }

    def common(self, x):
        x = F.relu(self.conv_1(x))
        x = self.drop2(F.relu(self.conv_2(x)))
        x = self.drop3(F.relu(self.conv_3(x)))
        x = self.drop4(F.relu(self.conv_4(x)))
        return x
    
    def head(self, x, action):
        x = self.common(x)
        layer = self.action_to_layer[action]
        return F.relu(layer(x))

    def forward(self, x):
        # mainly for visualization
        x = self.common(x)
        layer = self.conv_shoot
        return F.relu(layer(x))
