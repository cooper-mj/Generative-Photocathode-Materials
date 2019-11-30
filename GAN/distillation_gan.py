#!/usr/bin/env python

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from discriminator import Discriminator
from gan import load_dataset, batch_dataset, train
from generator import Generator
from utils import load_dataset
from logger_utils import Logger
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms, datasets


# ==============================================================================
# Distillation GAN
# ==============================================================================
def get_training_partitions():
    """
    Generates column-partitioned training sets for various GANs
    """
    atomic_X = X[:]  # TODO: partition into the atomic number features
    locations_X = X[:]  # TODO: partition into connection features
    other_X = X[:]

    return training_partitions


def init_population(partitions):
    """
    Initializes a population given the initial training partitions
    """
    return []


def mutate(population):
    """
    Trains a GAN for each population element
    """
    pass


def select_fittest(X, num_batches, k):
    """
    Runs GAN on particles dataset k times, and selects the most fit teacher network
    """
    fittest_model, highest_fitness = None, 0
    for i in range(k):
        G, D, particle_samples, evaluations = train(X, num_batches, args.num_particle_samples)
        MLE_emittance = torch.mean(evaluations)
        if MLE_emittance > highest_fitness:
            fittest_model = (G, D)
            highest_fitness = MLE_emittance

    print('Over %d runs, found a fittest GAN teacher with an MLE emittance of %.2f' % (k, highest_fitness))
    return fittest_model, highest_fitness


def crossover(pol1, pol2):
    """
    Genetic operator the crosses together two GANs trained separately
    """
    pass


def dagger_distillation(crossover_pol):
    """
    Uses imitation learning DAGGER to distill a crossover_policy into a child
    policy with the same architecture as the original GAN
    """
    pass


def breed(parents):
    """
    Runs crossover and dagger_distillation on pairs of fit parents
    """
    pass


def train_KD_GAN(X, Y, num_batches, k, r):
    """
    Trains a student GAN network from a teacher GAN selected from k GAN runs and r epochs
        Specifically for our project, we would like to experiment with
        generating our atomic numbers and connections separately, to do this,
        we distill a GAN trained on atomic numbers and a GAN trained on connections
        into a binary GAN policy and train a student using a framework similar
        to GPO (https://arxiv.org/pdf/1711.01012.pdf)

    # TODO create a population
    """
    training_partitions = get_training_partitions(X)
    population = init_population(training_partitions)
    epoch = 0

    while epoch < r and len(population) > 1:
        population = mutate(population)
        parents = select_fittest(population)
        population = breed(parents)

    return population[0]

    # fittest_atomic_GAN, highest_atomic_fitness = select_fittest(atomic_X, num_batches, k)
    # fittest_connections_GAN, highest_connections_fitness = select_fittest(connections_X, num_batches, k)
    # fittest_others_GAN, highest_others_fitness = select_fittest(other_X, num_batches, k)

    # binary_crossover_GAN, crossover_fitness = crossover(fittest_atomic_GAN, fittest_connections_GAN)
    # student_GAN, student_fitness = dagger_distillation(binary_crossover_GAN)
    # return student_GAN, student_fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--latent', type=int, default=8, help="Latent representation size")
    parser.add_argument('--g_input_size', type=int, default=8, help="Random noise dimension coming into generator, per output vector")
    parser.add_argument('--g_hidden_size', type=int, default=32, help="Generator complexity")
    parser.add_argument('--g_output_size', type=int, default=71, help="Size of generator output vector")

    parser.add_argument('--d_input_size', type=int, default=71, help="Minibatch size - cardinality of distributions (change)")
    parser.add_argument('--d_hidden_size', type=int, default=32, help="Discriminator complexity")
    parser.add_argument('--d_output_size', type=int, default=1, help="Single dimension for real vs fake classification")

    parser.add_argument('--p', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--d_learning_rate', type=float, default=1e-3)
    parser.add_argument('--g_learning_rate', type=float, default=1e-3)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)

    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--print_interval', type=int, default=5)

    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--k', type=int, default=5, help="Number of GANs to select teacher from")
    parser.add_argument('--num_particle_samples', type=int, default=100, help="Number of sample particles to aggregate fitness estimate over")
    parser.add_argument('--r_epochs', type=int, default=3, help="Number of epochs of GPO")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, X, Y = load_dataset("../unit_cell_data_16.csv")
    X = batch_dataset(X)
    num_batches = len(X)

    student = train_KD_GAN(X, Y, num_batches, args.k, args.r_epochs)
