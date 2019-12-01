#!/usr/bin/env python

import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from discriminator import Discriminator
from gan import load_dataset, batch_dataset, train, gen_noise
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
    X = torch.tensor(X, dtype=torch.float32)
    other_X = X[:,1:7]
    atomic_X = X[:]  # TODO: partition into the atomic number features, if not col, then 0's
    locations_X = X[:]  # TODO: partition into connection features
    partitions = [atomic_X, locations_X, other_X]
    return partitions


def init_population(X, num_batches):
    """
    Initializes a population given the initial training partitions
    """
    partitions = get_training_partitions(X)
    generation = 0
    population = dict()
    for i, partition in enumerate(partitions):
        G, D, _, evaluations = train(
            partition,
            num_batches,
            args.num_particle_samples,
            set_args=args
        )
        MLE_emittance = torch.mean(evaluations)
        population['gen%dpartition%d' % (generation, i)] = {
            'generator': G,
            'discriminator': D,
            'emittance': MLE_emittance,
            'partition': partition
        }
    print('Initialized the population!\n')
    return population


def mutate(population, num_batches, generation):
    """
    Trains a GAN for each population element
    """
    population = dict()
    i = 0
    for label, map in population.items():
        G, D, _, evaluations = train(
            map['partition'],
            num_batches,
            args.num_particle_samples,
            G=map['generator'],
            D=map['discriminator'],
            set_args=args
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


def select_fittest(population, k):
    """
    Select k fittest GANs
    TODO: debug this function
    """
    if len(population) <= k:
        sorted_population = sorted(population.items(), key=lambda kv: kv[1]['emittance'])[:k]  # results in a list of tuples
    else:
        sorted_population = population
    parents = {tuple[0]: tuple[1] for tuple in sorted_population}
    return parents


def crossover(pol1, pol2, pol3):
    """
    Genetic operator that crosses together two GANs trained separately
    Specifically for our case, we note our child GANS train on different parts
    of the dataset. Hence, we combine to create a new good dataset extension
    to match with a new student GAN
    """
    G_1 = pol1['generator']
    p_1 = pol1['partition']

    G_2 = pol2['generator']
    p_2 = pol2['partition']

    G_3 = pol3['generator']
    p_3 = pol3['partition']

    joint_partition = torch.cat((p_1, p_2, p_3), dim=0)
    test_noise_1 = gen_noise(args.crossover_samples, args.latent)
    test_noise_2 = gen_noise(args.crossover_samples, args.latent)
    test_noise_3 = gen_noise(args.crossover_samples, args.latent)

    fake_data_1 = G_1(d_noise).detach()
    fake_data_2 = G_2(d_noise).detach()
    fake_data_3 = G_3(d_noise).detach()

    # joint_fake_data = we need to join together the data that is being generated using the correct column indexing

    # For each generated feature col, choose the col that maximizes according to NN_eval (stochastic gradient descent)
        # After this, we already have a sub-GAN which is SGD optimized GAN combination
        # This is what Michael was originally interested in investigating
        # We can add this to population if it actually does better

    # Now that we have a new dataset, train a new GAN on it for imitation learning GAN. This takes it a step further by allowing multiple epochs of GPO
    pass


def breed(parents, population):
    """
    Runs crossover and dagger_distillation on pairs of fit parents
    """
    children = dict()

    triplets = list(itertools.combinations(parents.items(), 3))
    while len(population) - len(triplets) > 0:
        triplets.append(random.choice(triplets))
    if len(triplets) - len(population) > 0:
        triplets = random.sample(triplets, len(population))

    w = 0
    for p1, p2, p3 in triplets:
        G, D, _, evaluations = crossover(p1, p2, p3)
        MLE_emittance = torch.mean(evaluations)

        children['cross%d' % w] = {
            'generator': G,
            'discriminator': D,
            'emittance': MLE_emittance,
            'partition': partition
        }
        w += 1
    return children


def train_GPO_GAN(X, Y, num_batches, k, r):
    """
    Trains a student GAN network from a teacher GAN selected from population and r epochs
        Specifically for our project, we would like to experiment with
        generating our atomic numbers and connections separately, to do this,
        we distill a GAN trained on atomic numbers and a GAN trained on connections
        into a binary GAN policy and train a student using a framework similar
        to GPO (https://arxiv.org/pdf/1711.01012.pdf)

    We have modified this framework specifically to train GANs on various parts
    of the dataset separately.
    """
    population = init_population(X, num_batches)
    epoch = 0

    while epoch < r and len(population) > 1:
        if epoch > 0:
            population = mutate(population, num_batches, epoch)
        parents = select_k_fittest(population, k)
        population = breed(parents, population)
        epoch += 1

    # Run a final training on the resultant child to ensure training on full dataset
    student = select_k_fittest(population, 1)  # probably bug, need to isolate a single particle rather than a dict
    return train(
        X,
        num_batches,
        args.num_particle_samples,
        student['generator'],
        student['discriminator'],
        set_args=args
    )


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

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--print_interval', type=int, default=10)

    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--k', type=int, default=5, help="Number of GANs to select teacher from")
    parser.add_argument('--num_particle_samples', type=int, default=100, help="Number of sample particles to aggregate fitness estimate over")
    parser.add_argument('--r_epochs', type=int, default=3, help="Number of epochs of GPO")

    parser.add_argument('--crossover_samples', type=int, default=1000, help="number of samples for crossover")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, X, Y = load_dataset("../unit_cell_data_16.csv")
    X = batch_dataset(X, args.batch_size)
    num_batches = len(X)

    _, _, _, evaluations = train_GPO_GAN(X, Y, num_batches, args.k, args.r_epochs)
