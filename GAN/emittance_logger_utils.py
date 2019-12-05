import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

'''
    TensorBoard Data will be stored in './runs' path
'''

class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

# need: emittance mean and std; optional: graphical form of KNN also (mean and std)
#def log(self, d_error, g_error, epoch, n_batch, num_batches):
    def log(self, epoch, emittance_mean, emittance_std):

        step = Logger._step(epoch)
        self.writer.add_scalar(
            '{}/emittance_mean'.format(self.comment), emittance_mean, step)
        self.writer.add_scalar(
            '{}/emittance_std'.format(self.comment), emittance_std, step)

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch):
        return epoch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
