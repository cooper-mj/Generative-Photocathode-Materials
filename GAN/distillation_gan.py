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
def get_training_partitions(X):
    """
    Generates column-partitioned training sets for various GANs
    # TODO: for an extension, we can sample a number of random datasets
    """
    atomic_X = X[:]  # TODO: partition into the atomic number features
    locations_X = X[:]  # TODO: partition into connection features
    other_X = X[:]
    partions = [atomic_X, locations_X, other_X]
    return partitions


def init_population(X, num_batches):
    """
    Initializes a population given the initial training partitions
    """
    partitions = get_training_partitions(X)
    generation = 0
    population = map()
    for i, partition in enumerate(partitions):
        G, D, _, evaluations = train(
            partition,
            num_batches,
            args.num_particle_samples
        )
        MLE_emittance = torch.mean(evaluations)
        population['gen%dpartition%d' % (generation, i)] = {
            'generator': G,
            'discriminator': D,
            'emittance': MLE_emittance,
            'partition': partition
        }
    return population


def mutate(population, num_batches, generation):
    """
    Trains a GAN for each population element
    """
    population = map()
    i = 0
    for label, map in population.items():
        G, D, _, evaluations = train(
            map['partition'],
            num_batches,
            args.num_particle_samples,
            G=map['generator'],
            D=map['discriminator']
        )
        MLE_emittance = torch.mean(evaluations)
        population['gen%dpartition%d' % (generation, i)] = {
            'generator': G,
            'discriminator': D,
            'emittance': MLE_emittance,
            'partition': partition
        }
        i += 1
    return population


def select_fittest(X, num_batches, k):
    """
    Select k fittest GANs
    """
    # sort by map['emittance']
    # return top k


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
    Trains a student GAN network from a teacher GAN selected from population and r epochs
        Specifically for our project, we would like to experiment with
        generating our atomic numbers and connections separately, to do this,
        we distill a GAN trained on atomic numbers and a GAN trained on connections
        into a binary GAN policy and train a student using a framework similar
        to GPO (https://arxiv.org/pdf/1711.01012.pdf)
    """
    population = init_population(X, num_batches)
    epoch = 0

    while epoch < r and len(population) > 1:
        if epoch > 0:
            population = mutate(population, num_batches, epoch)
        parents = select_k_fittest(population, k)
        population = breed(parents)
        epoch += 1

    # Run a final training on the resultant child to ensure training on full dataset
    student = population[0]
    student = train(
        X,
        num_batches,
        args.num_particle_samples,
        student['generator'],
        student['discriminator']
    )

    return student


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
