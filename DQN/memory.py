import numpy as np
from collections import deque
import random


class MemoryUniform():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        """
        experience : 5-tuple representing a transaction (state, action, reward, next_state, done)
                     - state      : 4D-tensor (batch, motion, image)
                     - action     : 2D-tensor (batch, one_hot_encoded_action)
                     - reward     : 1D-tensor (batch,)
                     - next_state : 4D-tensor (batch, motion, image)
                     - done       : 1D-tensor (batch,)
        """
        
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)