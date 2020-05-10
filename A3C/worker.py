import numpy as np
import torch.multiprocessing as mp

from models import A2CNet
from settings import *
from utils import *


class Worker(mp.Process):
    def __init__(self, possible_actions, global_net, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.possible_actions = possible_actions
        self.name = 'w%02i' % name
        self.global_ep, self.global_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_net, self.opt = global_net, opt
        self.local_net = A2CNet(stack_size=STACK_SIZE, actions=len(self.possible_actions)) # local network
        if USE_GPU:
            self.local_net.cuda()

    def run(self):
        game, _ = create_environment(scenario=SCENARIO)
        total_step = 1
        episode = 1
        last_frames = None
        while self.global_ep.value < TOTAL_EPISODES:
            game.new_episode()
            episode_length = 1
            losses = []
            state, last_frames = stack_frames(last_frames, get_state(game), is_new_episode=True, maxlen=STACK_SIZE)
            buffer_s, buffer_a, buffer_r = [], [], []
            total_reward = 0.
            while True:
                buffer_s.append(state)
                action = self.local_net.choose_action(state)
                r = game.make_action(self.possible_actions[action], FRAME_SKIP)
                done = game.is_episode_finished()
                if not done:
                    state, last_frames = stack_frames(
                        last_frames,
                        get_state(game),
                        is_new_episode=True,
                        maxlen=STACK_SIZE,
                    )
                else:
                    r = -1
                total_reward += r
                buffer_a.append(action)
                buffer_r.append(r)

                if total_step % BATCH_SIZE == 0 or done:  # update global and assign to local net
                    # sync
                    loss = push_and_pull(
                        self.opt,
                        self.local_net,
                        self.global_net,
                        done,
                        state,
                        buffer_s,
                        buffer_a,
                        buffer_r,
                        GAMMA,
                    )
                    losses.append(loss)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        self.log_episode(game, total_reward, episode_length, np.array(losses).mean())
                        episode += 1
                        break
                total_step += 1
                episode_length += 1
        self.res_queue.put(None)

    def log_episode(self, game, episode_reward, episode_length, mean_loss):
        with self.global_ep.get_lock():
            with self.global_ep_r.get_lock():
                self.global_ep.value += 1
                self.global_ep_r.value = episode_reward
                episode = self.global_ep.value
                reward = self.global_ep_r.value
        vars = {
            'Game/Kills': game.get_game_variable(KILLCOUNT),
            'Game/Ammo': game.get_game_variable(AMMO2),
            'Game/Length': episode_length,
            'Train/Reward': reward,
            'Train/Loss': mean_loss,
        }
        self.res_queue.put(vars)
        print('{}: episode={}, reward={}'.format(self.name, episode, reward))


