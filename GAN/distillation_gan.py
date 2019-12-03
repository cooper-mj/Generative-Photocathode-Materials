#!/usr/bin/env python

import argparse
import itertools
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pk

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

    other_idx = [i for i in range(0,7)]
    other_X = X[:,:,other_idx]

    atomic_idx = [i for i in range(7,71,4)]
    atomic_X = X[:,:,atomic_idx]

    locations_idx = [i for i in range(8,71) if i % 4 != 3]
    locations_X = X[:,:,locations_idx]

    return [(other_X, other_idx), (atomic_X, atomic_idx), (locations_X, locations_idx)]


def init_population(X, num_batches):
    """
    Initializes a population given the initial training partitions
    """
    partitions = get_training_partitions(X)
    generation = 0
    population = dict()
    for i, partition in enumerate(partitions):
        spec_args = args
        spec_args.g_input_size = args.latent
        spec_args.g_output_size = len(partition[1])
        spec_args.g_hidden_size = int(math.ceil(spec_args.g_output_size / 2))
        spec_args.d_input_size = len(partition[1])
        spec_args.d_hidden_size = int(math.ceil(spec_args.d_input_size / 2))

        G, D, _, evaluations = train(
            partition[0],
            num_batches,
            args.num_particle_samples,
            set_args=spec_args,
            train_cols=partition[1]
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
    i = 0
    new_population = dict()
    for label, map in population.items():
        G, D, _, evaluations = train(
            map['partition'][0],
            num_batches,
            args.num_particle_samples,
            G=map['generator'],
            D=map['discriminator'],
            set_args=args,
            train_cols=map['partition'][1]
        )
        MLE_emittance = torch.mean(evaluations)
        new_population['gen%dpartition%d' % (generation, i)] = {
            'generator': G,
            'discriminator': D,
            'emittance': MLE_emittance,
            'partition': map['partition']
        }
        i += 1
    return new_population


def select_k_fittest(population, k):
    """
    Select k fittest GANs
    TODO: debug this function
    """
    if len(population) >= k:
        return sorted(population.values(), key=lambda v: v['emittance'])[:k]  # results in a list of tuples
    return population


def reshape_generator_output(pol1, pol2, pol3):
    """
    Reshapes the output of generators that do not match the original output size
    """
    G_1 = pol1['generator']
    p_1 = pol1['partition']

    G_2 = pol2['generator']
    p_2 = pol2['partition']

    G_3 = pol3['generator']
    p_3 = pol3['partition']

    test_noise_1 = gen_noise(args.crossover_samples, args.latent)
    test_noise_2 = gen_noise(args.crossover_samples, args.latent)
    test_noise_3 = gen_noise(args.crossover_samples, args.latent)

    fake_data_1 = G_1(test_noise_1).detach()  # make sure when we create a gan we can set its output shape (critical!)
    fake_data_2 = G_2(test_noise_2).detach()
    fake_data_3 = G_3(test_noise_3).detach()

    # Format back into their appropriate columns
    d_1 = torch.zeros(args.crossover_samples, 71)
    d_2 = torch.zeros(args.crossover_samples, 71)
    d_3 = torch.zeros(args.crossover_samples, 71)

    d_1[:,p_1[1]] = fake_data_1
    d_2[:,p_2[1]] = fake_data_2
    d_3[:,p_3[1]] = fake_data_3

    # Also format the datasets back into their appropriate columns
    jp_1 = torch.zeros(p_1[0].shape[0], p_1[0].shape[1], 71)
    jp_2 = torch.zeros(p_2[0].shape[0], p_2[0].shape[1], 71)
    jp_3 = torch.zeros(p_3[0].shape[0], p_3[0].shape[1], 71)

    jp_1[:,:,p_1[1]] = p_1[0]
    jp_2[:,:,p_2[1]] = p_2[0]
    jp_3[:,:,p_3[1]] = p_3[0]

    joint_partition = jp_1.add_(jp_2.add_(jp_3))
    return joint_partition, d_1, d_2, d_3, p_1, p_2, p_3


def sample_top_paths(joint_partition, gen_partition, clf):
    """
    Samples the top 50% of training samples from GPO and the previous dataset

        # Commented out code prints what the top particle representations are
        # for particle in ind:
        #     print(dataset_particles[ind,:])
    """
    dataset_particles = joint_partition.view(-1, 71)
    dataset_particles = torch.cat((dataset_particles, gen_partition), dim=0)
    percentile_index = math.floor(dataset_particles.shape[0]/2)

    dataset_particles = dataset_particles.detach().numpy()
    prediction = torch.tensor(clf.predict(dataset_particles), dtype=torch.float32)
    res, ind = prediction.topk(percentile_index, largest=False)

    # There may be a bug later where this fails. That would be because i did batching jankily long ago, fix it
    top_partition = batch_dataset(dataset_particles[ind,:], args.batch_size)
    return torch.tensor(top_partition, dtype=torch.float32)


def crossover(pol1, pol2, pol3):
    """
    Genetic operator that crosses together two GANs trained separately
    Specifically for our case, we note our child GANS train on different parts
    of the dataset. Hence, we combine to create a new good dataset extension
    to match with a new student GAN.

    We choose the crossover to operate on 3 GANs for the initial combination of the dataset
    """
    joint_partition, d_1, d_2, d_3, p_1, p_2, p_3 = reshape_generator_output(pol1, pol2, pol3)

    # Naive approach sequential greedy maximization
    file = open('NN_evaluator.sav', 'rb')
    clf = pk.load(file)

    gen_partition = torch.zeros(d_1.shape[0], 71, dtype=torch.float32)
    for col in range(71):
        grad_d_1 = gen_partition.clone()
        grad_d_2 = gen_partition.clone()
        grad_d_3 = gen_partition.clone()

        grad_d_1[:,col] = d_1[:,col]
        grad_d_2[:,col] = d_2[:,col]
        grad_d_3[:,col] = d_3[:,col]

        d_1_pred = torch.tensor(clf.predict((grad_d_1).detach().numpy()), dtype=torch.float32)
        d_2_pred = torch.tensor(clf.predict((grad_d_2).detach().numpy()), dtype=torch.float32)
        d_3_pred = torch.tensor(clf.predict((grad_d_3).detach().numpy()), dtype=torch.float32)

        e_1 = torch.mean(d_1_pred)
        e_2 = torch.mean(d_2_pred)
        e_3 = torch.mean(d_3_pred)

        if e_1 < e_2 and e_1 < e_3:
            gen_partition = grad_d_1
        elif e_2 < e_1 and e_2 < e_3:
            gen_partition = grad_d_2
        else:
            gen_partition = grad_d_3
    # ======================================================================== #
    # For each generated feature col, choose the col that maximizes according to NN_eval (stochastic gradient descent or adam)
        # After this, we already have a sub-GAN which is SGD optimized GAN combination
        # This is what Michael was originally interested in investigating
        # The only problem is this is not a single GAN, so we cannot add it to a population and iterate
        # Print the results (i.e. mean emittance for the crossover) for sure!
    # ======================================================================== #
    top_partition = sample_top_paths(joint_partition, gen_partition, clf)

    # Now that we have a new dataset, train a new GAN on it for imitation learning GAN.
    spec_args = args
    spec_args.g_input_size = args.latent
    spec_args.g_output_size = top_partition.shape[2]
    spec_args.g_hidden_size = int(math.ceil(spec_args.g_output_size / 2))
    spec_args.d_input_size = top_partition.shape[2]
    spec_args.d_hidden_size = int(math.ceil(spec_args.d_input_size / 2))
    partition_idx = list(set(p_1[1] + p_2[1] + p_3[1]))

    return train(
        top_partition,
        num_batches,
        args.num_particle_samples,
        set_args=spec_args,
        train_cols=partition_idx
    ), (top_partition, partition_idx)


def breed(parents, population):
    """
    Runs crossover and 'dagger'_distillation on pairs of fit parents
    """
    children = dict()

    triplets = list(itertools.combinations(parents.values(), 3))
    while len(population) - len(triplets) > 0:
        triplets.append(random.choice(triplets))
    if len(triplets) - len(population) > 0:
        triplets = random.sample(triplets, len(population))

    w = 0
    gen_best_score = None
    for p1, p2, p3 in triplets:
        (G, D, _, evaluations), partition = crossover(p1, p2, p3)
        MLE_emittance = torch.mean(evaluations)
        if not gen_best_score or MLE_emittance < gen_best_score:
            gen_best_score = MLE_emittance

        children['cross%d' % w] = {
            'generator': G,
            'discriminator': D,
            'emittance': MLE_emittance,
            'partition': partition
        }
        w += 1
    print('The best emittance score for this generation is %2f' % gen_best_score)
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
    student = select_k_fittest(population, 1)[0]  # probably bug, need to isolate a single particle rather than a dict
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
    parser.add_argument('--print_interval', type=int, default=200)

    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--k', type=int, default=5, help="Number of GANs to select teacher from")
    parser.add_argument('--num_particle_samples', type=int, default=100, help="Number of sample particles to aggregate fitness estimate over")
    parser.add_argument('--r_epochs', type=int, default=3, help="Number of epochs of GPO")

    parser.add_argument('--crossover_samples', type=int, default=1000, help="number of samples for crossover")

    parser.add_argument('--loss_fn', type=str, default='dflt') # Loss defaults to whatever is defined in the train function - currently nn.BCELoss()

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, X, Y = load_dataset("../unit_cell_data_16.csv")
    X = batch_dataset(X, args.batch_size)
    num_batches = len(X)

    _, _, _, evaluations = train_GPO_GAN(X, Y, num_batches, args.k, args.r_epochs)
    print('Final disciple has evaluation score of %2f' % torch.mean(evaluations))
