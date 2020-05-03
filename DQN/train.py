import argparse
from agent import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train options')

    parser.add_argument('--scenario', type=str, default='basic', metavar='S', help="scenario to use, either basic or deadly_corridor")
    parser.add_argument('--window', type=int, default=0, metavar='WIN', help="0: don't render screen | 1: render screen")
    parser.add_argument('--resize', type=tuple, default=(120, 160), metavar='RES', help="Size of the resized frame")
    parser.add_argument('--stack_size', type=int, default=4, metavar='SS', help="Number of frames to stack to create motion")
    parser.add_argument('--explore_start', type=float, default=1., metavar='EI', help="Initial exploration probability")
    parser.add_argument('--explore_stop', type=float, default=0.01, metavar='EL', help="Final exploration probability")
    parser.add_argument('--decay_rate', type=float, default=1e-3, metavar='DR', help="Decay rate of exploration probability")
    parser.add_argument('--memory_size', type=int, default=1000, metavar='MS', help="Size of the experience replay buffer")
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help="Batch size")
    parser.add_argument('--gamma', type=float, default=.99, metavar='GAMMA', help="Discounting rate")
    parser.add_argument('--total_episodes', type=int, default=500, metavar='EPOCHS', help="Number of training episodes")
    parser.add_argument('--pretrain', type=int, default=100, metavar='PRE',
                        help="number of initial experiences to put in the replay buffer")
    parser.add_argument('--frame_skip', type=int, default=4, metavar='FS', help="the number of frames to repeat the action on")
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help="The learning rate")
    parser.add_argument('--max_tau', type=int, default=100, metavar='LR', help="Number of steps to performe double q-learning update")
    parser.add_argument('--save_freq', type=int, default=50, metavar='SFQ', help="Number of episodes to save model weights")
    parser.add_argument('--logfile', type=str, default=None, metavar='LOGFILE', help='Filename for tensorboard logs and model weights')
    parser.add_argument('--load_weights', type=str, default=None, metavar='LOAD', help="Path to the weights we want to load (if we want)")

    args = parser.parse_args()
    game, possible_actions = create_environment(scenario=args.scenario, window=args.window)
    agent = Agent(possible_actions, args.scenario, max_size=args.memory_size, stack_size=args.stack_size,
                  batch_size=args.batch_size, resize=args.resize)
    agent.train(game, total_episodes=args.total_episodes, pretrain=args.pretrain, frame_skip=args.frame_skip, lr=args.lr,
                max_tau=args.max_tau, explore_start=args.explore_start, explore_stop=args.explore_stop, decay_rate=args.decay_rate,
                gamma=args.gamma, save_freq=args.save_freq, load_weights=args.load_weights, logfile=args.logfile)
