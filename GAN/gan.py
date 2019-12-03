#!/usr/bin/env python

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pk

from discriminator import Discriminator
from generator import Generator
from utils import load_dataset
from logger_utils import Logger
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms, datasets

from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

# ==============================================================================
# Nonsaturating Loss Function
# ==============================================================================
def nonsaturating_loss(G, D, fake_data, real_data):

    d_loss = -torch.mean(nn.functional.logsigmoid(D(real_data))) - torch.mean(torch.log(1 - torch.sigmoid(D(fake_data))))

    g_loss = -torch.mean(nn.functional.logsigmoid(D(fake_data)))
    return d_loss, g_loss

# ==============================================================================
# Data sampler for Particles
# ==============================================================================
def batch_dataset(x, batch_size):
    """
    partitions dataset x into args.batch_size batches
    TODO: ensure that x.shape[0] is divisible by batch_size so no leftovers
    """
    size_modulo = len(x) % batch_size  # hack to ensure data is batches successfully
    if size_modulo != 0:
        x = x[:-size_modulo]
    partitioned = np.split(x, batch_size)
    return partitioned


def gen_noise(sample_size, latent):
    """
    Sample a sample_size noise tensors, each with latent elements
    """
    return Variable(torch.randn(sample_size, latent))

# ==============================================================================
# Train loop
# ==============================================================================
def get_optimizers(args):
    """
    Note, our latent representation vector for our 72-dimensional particles is
    dimension 8
    """
    # Create a generator which can map a latent vector size 8 to 72
    G = Generator(
        input_size=args.g_input_size,
        hidden_size=args.g_hidden_size,
        output_size=args.g_output_size,
        p=args.p
    )
    # Create a discriminator which can turn 72-dimensional particle to Binary
    # prediction
    D = Discriminator(
        input_size=args.d_input_size,
        hidden_size=args.d_hidden_size,
        output_size=args.d_output_size,
        p=args.p,
        dropout=args.dropout
    )

    # Choose an optimizer
    if args.optim == 'Adam':
        d_optimizer = optim.Adam(D.parameters(), lr=args.d_learning_rate)
        g_optimizer = optim.Adam(G.parameters(), lr=args.g_learning_rate)
    else:
        d_optimizer = optim.SGD(D.parameters(), lr=args.d_learning_rate)
        g_optimizer = optim.SGD(G.parameters(), lr=args.g_learning_rate, momentum=args.sgd_momentum)
    return G, D, d_optimizer, g_optimizer

def train_discriminator(G, D, d_optimizer, loss, real_data, fake_data, loss_fn):
    """
    Trains the Discriminator for one step
    """
    N = real_data.size(0)
    d_optimizer.zero_grad()

    # Train D on real data
    pred_real = D(real_data)
    if loss_fn == "dflt":
        error_real = loss(pred_real, Variable(torch.ones(N, 1)))
    elif loss_fn == "nonsaturating":
        error_real, _ = nonsaturating_loss(G, D, real_data, fake_data)

    error_real.backward()

    # Train on fake data
    pred_fake = D(fake_data)
    if loss_fn == "dflt":
        error_fake = loss(pred_fake, Variable(torch.ones(N, 1)))
    elif loss_fn == "nonsaturating":
        _, error_fake = nonsaturating_loss(G, D, real_data, fake_data)

    error_fake.backward()

    d_optimizer.step()
    return error_real + error_fake, pred_real, pred_fake

def train_generator(G, D, g_optimizer, loss, real_data, fake_data, loss_fn):
    """
    Trains the Generator for one step
    """

    N = fake_data.size(0)
    g_optimizer.zero_grad()

    # predict against fake data
    pred = D(fake_data)

    if loss_fn == "dflt":
        error = loss(pred, Variable(torch.ones(N, 1)))
    elif loss_fn == "nonsaturating":
        _, error = nonsaturating_loss(G, D, real_data, fake_data)

    error.backward()
    g_optimizer.step()
    return error

def train(X, num_batches, num_particle_samples=100, G=None, D=None, set_args=None, train_cols=None):
    if set_args:
        args = set_args
    logger = Logger(model_name='GAN', data_name='Particles')
    loss = nn.BCELoss()  # Utilizing Binary Cross Entropy Loss
    if not G and not D:
        G, D, d_optimizer, g_optimizer = get_optimizers(args)
    else:
        _, _, d_optimizer, g_optimizer = get_optimizers(args)

    # Sample particles to examine progress
    test_noise = gen_noise(num_particle_samples, args.latent)

    for epoch in tqdm(range(args.num_epochs)):
        for n_batch, real_particle_batch in enumerate(X):
            real_particle_batch = torch.Tensor(real_particle_batch)
            N = real_particle_batch.shape[0] # This should be the batch size

            # Train the discriminator once
            real_data = Variable(real_particle_batch)  # gets a single row from the particles data
            d_noise = gen_noise(N, args.latent)  # generate a batch of noise vectors
            fake_data = G(d_noise).detach()

            total_d_error, d_pred_real, d_pred_fake = train_discriminator(G, D, d_optimizer, loss, real_data, fake_data, args.loss_fn)

            # Train the generator 5 times for every 1 time we train the discriminator
            for d_step in range(5):
                g_noise = gen_noise(N, args.latent)
                fake_data = G(g_noise)
                g_error = train_generator(G, D, g_optimizer, loss, real_data, fake_data, args.loss_fn)

            # Run logging to examine progress
            logger.log(total_d_error, g_error, epoch, n_batch, num_batches)
            # if (n_batch % args.print_interval) == 0:
            #     logger.display_status(
            #         epoch,
            #         args.num_epochs,
            #         n_batch,
            #         num_batches,
            #         total_d_error,
            #         g_error,
            #         d_pred_real,
            #         d_pred_fake
            #     )
    # Import the evaluator NN
    file = open('NN_evaluator.sav', 'rb')
    clf = pk.load(file)

    # Generate a test particle
    sample_particle = G(test_noise)
    if train_cols:
        d = torch.zeros(num_particle_samples, 71)
        d[:,train_cols] = sample_particle
        sample_particle = d
    # Evaluator predicts on that particle
    sample_particle = sample_particle.detach().numpy()
    prediction = torch.tensor(clf.predict(sample_particle), dtype=torch.float32)
    # Printout
    print("Generated Example Particles")
    print(sample_particle)
    print("Example Particle Predictions")
    print(prediction)
    return G, D, sample_particle, prediction


def local_parser():
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

    parser.add_argument('--loss_fn', type=str, default='dflt') # Loss defaults to whatever is defined in the train function - currently nn.BCELoss()

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = local_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, X, Y = load_dataset("../unit_cell_data_16.csv")
    X = batch_dataset(X, args.batch_size)
    num_batches = len(X)

    train(X, num_batches, set_args=args)
