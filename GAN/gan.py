#!/usr/bin/env python

# Generative Adversarial Networks (GAN) example in PyTorch. Tested with PyTorch 0.4.1, Python 3.6.7 (Nov 2018)
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from discriminator import Discriminator
from generator import Generator
from utils import load_dataset
from matplotlib import pyplot as plt
from torch.autograd import Variable

# Data params
data_mean = 4
data_stddev = 1.25

# ### Uncomment only one of these to define what data is actually sent to the Discriminator
(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
#(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)
#(name, preprocess, d_input_func) = ("Data and diffs", lambda data: decorate_with_diffs(data, 1.0), lambda x: x * 2)
#(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)

# ==============================================================================
# Data sampler for distributions example
# ==============================================================================

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

# ==============================================================================
# Data sampler for Particles
# ==============================================================================

def get_discriminator_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

# ==============================================================================
# Utilities
# ==============================================================================
def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def get_moments(d):
    # Return the first 4 moments of the data provided
    mean = torch.mean(d)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
    final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
    return final

def decorate_with_diffs(data, exponent, remove_raw_data=False):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    if remove_raw_data:
        return torch.cat([diffs], 1)
    else:
        return torch.cat([data, diffs], 1)

# ==============================================================================
# Train loop
# ==============================================================================
def get_optimizers():
    discriminator_activation_function = torch.sigmoid
    generator_activation_function = torch.tanh

    G = Generator(input_size=args.g_input_size,
                  hidden_size=args.g_hidden_size,
                  output_size=args.g_output_size,
                  f=generator_activation_function)
    D = Discriminator(input_size=d_input_func(args.d_input_size),
                      hidden_size=args.d_hidden_size,
                      output_size=args.d_output_size,
                      f=discriminator_activation_function)

    # Choose an optimizer
    if args.optim == 'Adam':
        d_optimizer = optim.Adam(D.parameters(), lr=args.d_learning_rate)
        g_optimizer = optim.Adam(G.parameters(), lr=args.g_learning_rate)
    else:
        d_optimizer = optim.SGD(D.parameters(), lr=args.d_learning_rate)
        g_optimizer = optim.SGD(G.parameters(), lr=args.g_learning_rate, momentum=args.sgd_momentum)
    return G, D, d_optimizer, g_optimizer


def train(x, y):
    dfe, dre, ge = 0, 0, 0
    d_real_data, d_fake_data, g_fake_data = None, None, None
    print_interval = 100

    d_sampler = get_distribution_sampler(data_mean, data_stddev)
    gi_sampler = get_generator_input_sampler()
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    G, D, d_optimizer, g_optimizer = get_optimizers()

    for epoch in range(args.num_epochs):
        for d_index in range(args.d_steps):
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real
            d_real_data = Variable(d_sampler(args.d_input_size))
            d_real_decision = D(preprocess(d_real_data))
            d_real_error = criterion(d_real_decision, Variable(torch.ones([1,1])))  # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params

            #  1B: Train D on fake
            d_gen_input = Variable(gi_sampler(args.minibatch_size, args.g_input_size))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(preprocess(d_fake_data.t()))
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1,1])))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

            dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

        for g_index in range(args.g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = Variable(gi_sampler(args.minibatch_size, args.g_input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(preprocess(g_fake_data.t()))
            g_error = criterion(dg_fake_decision, Variable(torch.ones([1,1])))  # Train G to pretend it's genuine

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
            ge = extract(g_error)[0]

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
                  (epoch, dre, dfe, ge, stats(extract(d_real_data)), stats(extract(d_fake_data))))

    print("Plotting the generated distribution...")
    values = extract(g_fake_data)
    print(" Values: %s" % (str(values)))
    plt.hist(values, bins=50)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Histogram of Generated Distribution')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--g_input_size', type=int, default=1, help="Random noise dimension coming into generator, per output vector")
    parser.add_argument('--g_hidden_size', type=int, default=5, help="Generator complexity")
    parser.add_argument('--g_output_size', type=int, default=1, help="Size of generator output vector")
    parser.add_argument('--d_input_size', type=int, default=50, help="Minibatch size - cardinality of distributions (change)")
    parser.add_argument('--d_hidden_size', type=int, default=10, help="Discriminator complexity")
    parser.add_argument('--d_output_size', type=int, default=1, help="Single dimension for real vs fake classification")
    parser.add_argument('--minibatch_size', type=int, default=50)

    parser.add_argument('--d_learning_rate', type=float, default=1e-3)
    parser.add_argument('--g_learning_rate', type=float, default=1e-3)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)

    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--d_steps', type=int, default=20)
    parser.add_argument('--g_steps', type=int, default=20)

    parser.add_argument('--optim', type=str, default='SGD')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, X, Y = load_dataset("../unit_cell_data_16.csv")
    print(X.shape)
    print(Y.shape)

    train(x=X, y=Y)
