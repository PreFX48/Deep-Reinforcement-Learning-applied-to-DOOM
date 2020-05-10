from datetime import datetime
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from models import A2CNet
from settings import *
from shared_adam import SharedAdam
from utils import create_environment
from worker import Worker

if __name__ == '__main__':
    mp.set_start_method('spawn')  # for CUDA support
    game, possible_actions = create_environment(scenario=SCENARIO)
    game.close()

    writer = SummaryWriter(log_dir='runs/{}'.format(datetime.now().strftime('%H:%M')))

    global_net = A2CNet(stack_size=STACK_SIZE, actions=len(possible_actions))
    global_net.share_memory()
    opt = SharedAdam(global_net.parameters(), lr=1e-4, betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    workers = [
        Worker(possible_actions, global_net, opt, global_ep, global_ep_r, res_queue, i+1)
        for i in range(WORKERS)
    ]
    for worker in workers:
        worker.start()

    global_step = 1
    while True:
        metricpoint = res_queue.get()
        if metricpoint is not None:
            for key, value in metricpoint.items():
                writer.add_scalar(key, value, global_step)
            global_step += 1
        else:
            break

    for worker in workers:
        worker.join()
