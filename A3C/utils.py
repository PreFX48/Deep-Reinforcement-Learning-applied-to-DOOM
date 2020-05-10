import torch
import torchvision.transforms as T

import numpy as np
import random
import time
from vizdoom import *
from models import *
from settings import *

from collections import deque

import warnings

warnings.filterwarnings('ignore')

# Boolean specifying whether GPUs are available or not.
use_cuda = torch.cuda.is_available()


def create_environment(scenario, window=False):
    game = DoomGame()
    game.set_window_visible(window)
    game.load_config("scenarios/{}.cfg".format(scenario))
    game.set_doom_scenario_path("scenarios/{}.wad".format(scenario))
    game.init()
    if scenario == 'basic':
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]
    elif scenario == 'deadly_corridor':
        possible_actions = np.identity(6, dtype=int).tolist()
    elif scenario == 'defend_the_center':
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]
    else:
        raise ValueError('Invalid scenario')
    return game, possible_actions

def get_state(game):
    state = game.get_state().screen_buffer
    return state[:, :, None]


def transforms(resize=(120, 160)):
    return T.Compose([T.ToPILImage(),
                      T.Resize(resize),
                      T.ToTensor()])


def stack_frames(stacked_frames, state, is_new_episode, maxlen=4, resize=(120, 160)):
    # Preprocess frame
    frame = transforms(resize)(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([frame[None] for i in range(maxlen)], maxlen=maxlen)  # We add a dimension for the batch
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame[None])  # We add a dimension for the batch
    stacked_state = torch.cat(tuple(stacked_frames), dim=1)
    return stacked_state, stacked_frames


def push_and_pull(opt, local_net, global_net, done, state, states, actions, rewards, gamma):
    if done:
        v_s_ = 0.  # terminal
    else:
        v_s_ = local_net.value(state).to('cpu').data.numpy()[0, 0]

    buffer_v_target = []
    for r in rewards[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = local_net.loss_func(
        torch.from_numpy(np.vstack(states)).to(DEVICE),
        torch.from_numpy(np.array(actions) if actions[0].dtype == np.int64 else np.vstack(actions)).to(DEVICE),
        torch.from_numpy(np.array(buffer_v_target)[:, None]).to(DEVICE)
    )

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
        global_param._grad = local_param.grad
    opt.step()

    # pull global parameters
    local_net.load_state_dict(global_net.state_dict())
    if USE_GPU:
        local_net.cuda()

    return loss.item()
