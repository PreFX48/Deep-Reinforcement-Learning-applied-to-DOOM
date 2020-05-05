import argparse
from agent import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train options')

    parser.add_argument('--scenario', type=str, default='basic', metavar='S', help="scenario to use, either basic or deadly_corridor")
    parser.add_argument('--window', type=int, default=0, metavar='WIN', help="0: don't render screen | 1: render screen")
    parser.add_argument('--resize', type=tuple, default=(120, 160), metavar='RES', help="Size of the resized frame")
    parser.add_argument('--stack_size', type=int, default=4, metavar='SS', help="Number of frames to stack to create motion")
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help="Batch size")
    parser.add_argument('--total_episodes', type=int, default=500, metavar='EPOCHS', help="Number of training episodes")
    parser.add_argument('--frame_skip', type=int, default=4, metavar='FS', help="the number of frames to repeat the action on")
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help="The learning rate")
    parser.add_argument('--save_freq', type=int, default=50, metavar='SFQ', help="Number of episodes to save model weights")
    parser.add_argument('--logfile', type=str, default=None, metavar='LOGFILE', help='Filename for tensorboard logs and model weights')
    parser.add_argument('--load_weights', type=str, default=None, metavar='LOAD', help="Path to the weights we want to load (if we want)")
    parser.add_argument('--ignore_trained_layers', type=str, default=None, metavar='IGN', help="Layers to exclude from loading")

    args = parser.parse_args()
    game, possible_actions = create_environment(scenario=args.scenario, window=args.window)
    agent = Agent(possible_actions, args.scenario, stack_size=args.stack_size, batch_size=args.batch_size, resize=args.resize)
    agent.train(game, total_episodes=args.total_episodes, frame_skip=args.frame_skip, lr=args.lr,
                save_freq=args.save_freq, load_weights=args.load_weights, ignore_trained_layers=args.ignore_trained_layers, logfile=args.logfile)
