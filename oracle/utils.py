import torch
import torchvision.transforms as T

import numpy as np
import random
import time
from vizdoom import *
from models import *
import matplotlib.pyplot as plt

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
    assert stacked_frames or frame
    if state is not None:
        frame = transforms(resize)(state)
        if is_new_episode:
            stacked_frames = deque([frame[None] for i in range(maxlen)], maxlen=maxlen)  # We add a dimension for the batch
        else:
            stacked_frames.append(frame[None])  # We add a dimension for the batch
    stacked_state = torch.cat(tuple(stacked_frames), dim=1)
    return stacked_state, stacked_frames


def action_vector_to_name(action):
    if action[0] == 1:
        action = 'right'
    elif action[1] == 1:
        action = 'shoot'
    elif action[2] == 1:
        action = 'right'
    else:
        action = 'UNKNOWN'
    return action


def draw_predictions(images):
    fig = plt.figure(figsize=(4.5, 2))
    ax = fig.add_subplot(1, 3, 1, xticks=[], yticks=[])
    plt.imshow(images[0][-1], cmap="Greys")
    ax.set_title('State')
    ax = fig.add_subplot(1, 3, 2, xticks=[], yticks=[])
    plt.imshow(images[1][-1], cmap="Greys")
    ax.set_title('Prediction')
    ax = fig.add_subplot(1, 3, 3, xticks=[], yticks=[])
    plt.imshow(images[2][-1], cmap="Greys")
    ax.set_title('Truth')
    return fig


def drop_incompatible_layers(model, state_dict, layers_to_ignore=None):
        """Drop layers from state_dict which are not present in the model, have different shape or are manually specified in layers_to_ignore"""
        model_state_dict = model.state_dict()

        unexpected = set(state_dict.keys()) - set(model_state_dict.keys())
        missing = set(model_state_dict.keys()) - set(state_dict.keys())
        ignored = set()
        incompatible = set()

        if layers_to_ignore:
            for layer in layers_to_ignore:
                if '{}.weight'.format(layer) in model_state_dict:
                    ignored.add('{}.weight'.format(layer))
                if '{}.bias'.format(layer) in model_state_dict:
                    ignored.add('{}.bias'.format(layer))

        for key in set(state_dict.keys()) & set(model_state_dict.keys()):
            if state_dict[key].shape != model_state_dict[key].shape:
                incompatible.add(key)

        for key in (unexpected | ignored | incompatible):
            del state_dict[key]

        if unexpected or missing or ignored or incompatible:
            print('WARNING: some weights were not loaded for the model. Unexpected: {}. Missing: {}. Ignored: {}. Incompatible: {}'.format(
                ','.join(unexpected) if unexpected else 'NONE',
                ','.join(missing) if missing else 'NONE',
                ','.join(ignored) if ignored else 'NONE',
                ','.join(incompatible) if incompatible else 'NONE',
            ))
        
