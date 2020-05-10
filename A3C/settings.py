import os
import torch
import torch.multiprocessing as mp

class NOT_PROVIDED:
    pass

def getenv(key, default=NOT_PROVIDED, transform=None):
    if default is not NOT_PROVIDED:
        value = os.environ.get(key, NOT_PROVIDED)
        if value is NOT_PROVIDED:
            value = default
        else:
            if transform is not None:
                value = transform(value)
    else:
        value = os.environ[key]
        if transform is not None:
            value = transform(value)
    return value


SCENARIO = getenv('SCENARIO', 'defend_the_center')
WORKERS = getenv('WORKERS', mp.cpu_count(), int)
FRAME_SKIP = getenv('FRAME_SKIP', 4, int)
STACK_SIZE = getenv('STACK_SIZE', 4, int)
TOTAL_EPISODES = getenv('TOTAL_EPISODES', NOT_PROVIDED, int)  # per worker
BATCH_SIZE = getenv('BATCH_SIZE', 5, int)
GAMMA = getenv('GAMMA', 0.9, float)

USE_GPU = (os.environ['USER'] != 'v-sopov')  # use gpu everywhere except for laptop
DEVICE = 'cuda' if USE_GPU else 'cpu'
