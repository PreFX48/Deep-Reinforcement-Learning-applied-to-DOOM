import torch
import torchvision.transforms as T

import numpy as np
import random
import time
from vizdoom import *
from models import *

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


def test_environment(weights, scenario, window=False, total_episodes=100, frame_skip=2, stack_size=4):
    game = DoomGame()
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_window_visible(window)

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

    model = DQNetwork(stack_size=stack_size, actions=len(possible_actions))
    if use_cuda:
        model.cuda()

    # Load the weights of the model
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)
    for i in range(total_episodes):
        game.new_episode()
        done = game.is_episode_finished()
        state = get_state(game)
        in_channels = model._modules['conv_1'].in_channels
        stacked_frames = deque([torch.zeros((120, 160), dtype=torch.int) for i in range(in_channels)], maxlen=in_channels)
        state, stacked_frames = stack_frames(stacked_frames, state, True, in_channels)
        while not done:
            if use_cuda:
                q = model(state.cuda())

            else:
                q = model(state)

            action = possible_actions[int(torch.max(q, 1)[1][0])]
            reward = game.make_action(action, frame_skip)
            done = game.is_episode_finished()
            if not done:
                state = get_state(game)
                state, stacked_frames = stack_frames(stacked_frames, state, False, in_channels)
            time.sleep(0.02)
        print("Total reward:", game.get_total_reward())
        time.sleep(0.1)
    game.close()


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
        # Stack the frames
        stacked_state = torch.cat(tuple(stacked_frames), dim=1)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame[None])  # We add a dimension for the batch
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = torch.cat(tuple(stacked_frames), dim=1)
    return stacked_state, stacked_frames


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, model, possible_actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
    else:
        if use_cuda:
            Qs = model.forward(state.cuda())
        else:
            Qs = model.forward(state)
        action = possible_actions[int(torch.max(Qs, 1)[1][0])]
    return action, explore_probability


def update_target(current_model, target_model):
    # Update the parameters of target_model with those of current_model
    target_model.load_state_dict(current_model.state_dict())


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
        
