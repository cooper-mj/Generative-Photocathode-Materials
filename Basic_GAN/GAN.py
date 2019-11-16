#
# CS236 Fall 2019-2020
# @jzzheng
#

import numpy as np
import torch

from discriminator import Discriminator
from generator import Generator
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

NUM_EPOCHS = 1
NUM_DISCRIMINATOR_STEPS = 1
NUM_GENERATOR_STEPS = 1

# ==============================================================================
# Rote Implementation
# ==============================================================================
def test_distribution_sampler(mu, sigma):
    """
    only used for rote implementation of GAN
    """
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))

def test_generator_sampler():
    """
    only used for rote implementation of GAN
    """
    return lambda m, n: torch.randn(m, n)

# ==============================================================================
# Particle Implemenetation
# ==============================================================================
def unpack_dataset(path):
    """
    @params: path (str) — The input path as a string
    @returns: train_loader, test_loader
    """
    pass



def train(train_set):
    """
    @params: train_set — Training set of examples
    """
    for epoch in NUM_EPOCHS:
        for d_step in NUM_DISCRIMINATOR_STEPS:
            pass
        for g_step in NUM_GENERATOR_STEPS:
            pass

if __name__ == '__main__':
    sample_path = 'hi'
    train_loader, test_loader = unpack_dataset(sample_path)
    train(train_loader)
