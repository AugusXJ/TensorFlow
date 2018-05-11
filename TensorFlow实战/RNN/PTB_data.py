import time
import numpy as np
import tensorflow as tf
import RNN.ptb.reader as reader

class PTBInput(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) -1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)

class PTBModel(object):
    def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

