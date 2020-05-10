from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

from settings import *


def conv2d_size_out(height_width, kernel_size, stride):
    # Compute the output dimensions when applying a convolutional layer. 
    height, width = height_width
    new_height = (height - (kernel_size - 1) - 1) // stride + 1
    new_width = (width - (kernel_size - 1) - 1) // stride + 1
    return new_height, new_width


class A2CNet(nn.Module):
    def __init__(self, w=120, h=160, stack_size=4, actions=3):
        super(A2CNet, self).__init__()

        # Conv Module
        self.conv_1 = nn.Conv2d(in_channels=stack_size, out_channels=16, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        height, width = conv2d_size_out(
            conv2d_size_out(
                (h, w),
                8, 4,
            ),
            4, 2,
        )
        linear_input_size = height * width * 32

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.policy_fc2 = nn.Linear(512, actions)
        self.value_fc2 = nn.Linear(512, 1)

    def common(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def value(self, x):
        x = self.common(x)
        value = self.value_fc2(x)
        return value

    def policy(self, x):
        x = self.common(x)
        return self.policy_fc2(x)

    def choose_action(self, state, softmax_dim=1):
        self.eval()
        logits = self.policy(state.to(DEVICE)).data
        prob = F.softmax(logits, dim=softmax_dim).data
        action = Categorical(prob).sample().to('cpu').numpy()[0]
        return action

    def loss_func(self, states, actions, values):
        self.train()
        logits, pred_values = self.policy(states), self.value(states)
        td = values - pred_values
        value_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        policy_loss = Categorical(probs).log_prob(actions) * td.detach().squeeze()
        total_loss = (value_loss - policy_loss).mean()
        return total_loss
