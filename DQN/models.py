import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(height_width, kernel_size, stride):
    # Compute the output dimensions when applying a convolutional layer. 
    height, width = height_width
    new_height = (height - (kernel_size - 1) - 1) // stride + 1
    new_width = (width - (kernel_size - 1) - 1) // stride + 1
    return new_height, new_width


class DQNetwork(nn.Module):
    def __init__(self, w=120, h=160, stack_size=4, actions=3):
        super(DQNetwork, self).__init__()

        # Conv Module
        self.conv_1 = nn.Conv2d(in_channels=stack_size, out_channels=16, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1)

        height, width = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(
                    (h, w),
                    8, 4,
                ),
                4, 2,
            ),
            4, 1,
        )
        linear_input_size = height * width * 64

        self.fc1 = nn.Linear(linear_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, actions)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)
