from collections import namedtuple
from datetime import datetime
import os
import random

from torch import optim
from tensorboardX import SummaryWriter

from utils import *
from models import *


METRICS_WINDOW = 10  # size of rolling window for metrics
IMAGES_PERIOD = 10  # show examples of predictions every N episodes


class Agent:

    def __init__(self, possible_actions, scenario, stack_size=4, batch_size=64, resize=(120, 160)):
        self.stack_size = stack_size
        self.possible_actions = possible_actions
        self.scenario = scenario
        self.batch_size = batch_size
        self.resize = resize

    def get_reward(self, variables_cur, variables_prev):
        r = 0
        if self.scenario == 'defend_the_center':
            r += variables_cur['kills'] - variables_prev['kills']
            if variables_cur['ammo'] < variables_prev['ammo']:
                r -= 0.1
            if variables_cur['health'] < variables_prev['health']:
                r -= 0.1
        elif self.scenario == 'deadly_corridor':
            r += (variables_cur['kills'] - variables_prev['kills']) * 5
            if variables_cur['ammo'] < variables_prev['ammo']:
                r -= 0.1
            if variables_cur['health'] < variables_prev['health']:
                r -= 1
        return r

    def train_batch(self, model, optimizers, action, batch):
        action = tuple(action)
        optimizer = optimizers[action]

        states, next_states = zip(*batch)
        states = torch.cat(states)
        next_states = torch.cat([transforms()(x)[None, :, :] for x in next_states])
        if torch.cuda.is_available():
            states = states.cuda()
            next_states = next_states.cuda()
        predictions = model.head(states, action)
        loss = F.mse_loss(predictions, next_states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        random_idx = random.randrange(len(batch))
        random_transition = (
            states[random_idx].cpu().numpy(),
            predictions[random_idx].cpu().detach().numpy(),
            next_states[random_idx].cpu().numpy(),
            )
        return loss.cpu().item(), random_transition

    def train(self, game, total_episodes=100, frame_skip=4, lr=1e-4, save_freq=50, load_weights=None, ignore_trained_layers=None, logfile=None):
        if logfile is None:
            logfile = '{}_{}'.format(datetime.now().strftime('%H:%M'), self.scenario)

        # Setting tensorboadX and variables of interest
        writer = SummaryWriter(log_dir='runs/{}'.format(logfile))
        # Metrics for tensorboard:
        losses_by_action = {tuple(action): [] for action in self.possible_actions}

        model = OracleNetwork(stack_size=self.stack_size)
        if use_cuda:
            for model in model_by_action.values():
                model.cuda()
        if load_weights:
            ignore_trained_layers = ignore_trained_layers.split(',') if ignore_trained_layers else []
            state_dict = torch.load(load_weights)
            drop_incompatible_layers(model, state_dict, layers_to_ignore=ignore_trained_layers)
            model.load_state_dict(state_dict, strict=False)

        samples_by_action = {tuple(action): [] for action in self.possible_actions}
        optimizer_by_action = {
            tuple(action): optim.Adam(model.parameters(), lr=lr)
            for action in self.possible_actions
        }
        transitions = []
        for episode in range(total_episodes):
            episode_rewards = []
            episode_losses_by_action = {tuple(action): [] for action in self.possible_actions}
            game.new_episode()
            variables_cur = {'kills': game.get_game_variable(KILLCOUNT), 'health': game.get_game_variable(HEALTH),
                             'ammo': game.get_game_variable(AMMO2)}
            variables_prev = variables_cur.copy()
            # Get 1st state
            done = game.is_episode_finished()
            state = get_state(game)
            stacked_frames = deque([torch.zeros(self.resize, dtype=torch.int) for i in range(self.stack_size)], maxlen=self.stack_size)
            state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size, self.resize)
            while (not done):
                action = random.choice(self.possible_actions)
                reward = game.make_action(action, frame_skip)
                # Update the game vaiables dictionnaries and get the reshaped reward
                variables_cur['kills'] = game.get_game_variable(KILLCOUNT)
                variables_cur['health'] = game.get_game_variable(HEALTH)
                variables_cur['ammo'] = game.get_game_variable(AMMO2)
                reward += self.get_reward(variables_cur, variables_prev)
                variables_prev = variables_cur.copy()
                # Check if the episode is done
                done = game.is_episode_finished()
                # Add the reward to total reward
                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros((240, 320), dtype='uint8')[:, :, None]
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resize)
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode+1))
                    # Saving metrics
                    for losses_action, episode_losses in episode_losses_by_action.items():
                        if episode_losses:
                            losses_by_action[losses_action].append(np.array(episode_losses).mean())
                    # Update writer
                    if (episode+1) % METRICS_WINDOW == 0:
                        for losses_action in self.possible_actions:
                            losses_action = tuple(losses_action)
                            writer.add_scalar('Train/Loss_{}'.format(action_vector_to_name(action)), np.array(losses_by_action[losses_action]).mean(), episode+1)
                            losses_by_action[losses_action] = []
                    if (episode+1) % IMAGES_PERIOD == 0 and transitions:
                        random_transition = random.choice(transitions)
                        transitions = []
                        writer.add_figure(
                            'Transitions/{}'.format(action_vector_to_name(random_transition[0])),
                            draw_predictions(random_transition[1]),
                            global_step=episode+1,
                        )
                else:
                    # Get the next state
                    next_state = get_state(game)
                    samples_by_action[tuple(action)].append((state, transforms()(next_state)))
                    next_stacked_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resize)
                    # If batch is full, learn on it
                    if len(samples_by_action[tuple(action)]) == self.batch_size:
                        batch_loss, transition = self.train_batch(model, optimizer_by_action, action, samples_by_action[tuple(action)])
                        transitions.append((tuple(action), transition))
                        episode_losses_by_action[tuple(action)].append(batch_loss)
                        samples_by_action[tuple(action)] = []
                    # Update state variable
                    state = next_stacked_state

            if (episode+1) % save_freq == 0:
                weights_dir = 'weights/' + logfile
                if not os.path.exists(weights_dir):
                    os.makedirs(weights_dir)
                model_file = '{}/{}.pth'.format(weights_dir, episode+1)
                torch.save(dqn_model.state_dict(), model_file)
                print('\nSaved model to ' + model_file)

        writer.close()
