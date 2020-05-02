import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=5, stride=2):
    # Compute the output dimension when applying a convolutional layer.
    return (size - (kernel_size - 1) - 1) // stride + 1


class DQNetwork(nn.Module):
    def __init__(self, w=120, h=160, stack_size=4, actions=3):
        super(DQNetwork, self).__init__()

        # Conv Module
        self.conv_1 = nn.Conv2d(in_channels=stack_size, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)

        convw = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2)  # width of last conv output
        convh = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2)  # height of last conv output
        linear_input_size = convw * convh * 64

        self.fc = nn.Linear(linear_input_size, 512)
        self.output = nn.Linear(512, actions)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.output(x)
