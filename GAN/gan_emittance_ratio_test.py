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
from emittance_logger_utils import Logger as ELogger
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms, datasets

from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

import matplotlib.pyplot as plt

NUM_EVALUATORS = 10

# ==============================================================================
# Elucidating Particle
# ==============================================================================
def get_particle_id_from_feature_tensor(features):
	ids, X_knn, Y_knn = load_dataset("../unit_cell_data_16.csv", threshold=float("inf"))
	for id in tqdm(range(ids.shape[0])):
		ft_xnn = torch.from_numpy(X_knn[id,:]).type(torch.FloatTensor)
		n_digits = 3
		rounded_ft_xnn = (ft_xnn * 10**n_digits).round() / (10**n_digits)
		rounded_features = (features * 10**n_digits).round() / (10**n_digits)
		# print(ft_xnn)
		# print(features)
		# break
		if torch.eq(rounded_ft_xnn, rounded_features).all():
		# if torch.eq(ft_xnn, features).all():
			print('The corresponding particle id is %d' % ids[id])
			predictions = torch.zeros(10)
			features = features.unsqueeze(0)
			for i in range(NUM_EVALUATORS):
				# Import the evaluator NN
				file = open('NN_evaluator_'+str(i)+'.sav', 'rb')
				clf = pk.load(file)
				# Using the evaluator NN, make a prediction on the generated particle
				predictions[i] = torch.tensor(clf.predict(features), dtype=torch.float32)
			emittance = torch.mean(predictions)
			print('Its emittance is: %2f' % emittance)
			return
	print('No corresponding particle id was found.')
	return

# ==============================================================================
# Nonsaturating Loss Function
# ==============================================================================
def nonsaturating_loss(G, D, fake_data, real_data):

	d_loss = -torch.mean(nn.functional.logsigmoid(D(real_data))) - torch.mean(torch.log(1 - torch.sigmoid(D(fake_data))))

	g_loss = -torch.mean(nn.functional.logsigmoid(D(fake_data)))
	return d_loss, g_loss

# ==============================================================================
# Wasserstein Loss Function
# ==============================================================================
def loss_wasserstein_gp(G, D, fake_data, real_data):
	'''
	Returns:
	- d_loss (torch.Tensor): wasserstein discriminator loss
	- g_loss (torch.Tensor): wasserstein generator loss
	'''

	batch_size = real_data.shape[0]
	lam = 10

	alpha = torch.rand(batch_size, requires_grad = True)
	alpha = torch.unsqueeze(alpha, 1)

	x_1 = fake_data
	x_2 = real_data
	x = alpha*x_1 + (torch.ones_like(alpha) - alpha)*x_2

	d_x = D(x)
	grad_d = torch.autograd.grad(d_x, x, grad_outputs= torch.ones_like(d_x), create_graph=True)[0]
	d_loss_third_term = lam*torch.mean((grad_d.norm(2) - torch.ones_like(grad_d))**2)
	d_g_z = D(fake_data)
	d_loss = torch.mean(d_g_z) - torch.mean(D(real_data)) + d_loss_third_term
	g_loss = -torch.mean(d_g_z)

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
# KNN functions
# ==============================================================================
# Function: knn
# -------------
# finds the k nearest neighbors in the dataset closest to the sample_particle
def knn(sample_particle, dataset, k):
	min_norm = float("inf")
	min_dict = {}
	for i in range(len(dataset)):
		x_real = dataset[i,:]
		l2_norm = torch.dist(sample_particle, x_real)
		min_dict[l2_norm] = x_real

	min_dists = sorted(min_dict.keys())[:k]
	k_nearest = torch.zeros_like(min_dict[min_dists[1]]) # if you think of a better way to init might do that instead
	inds = []
	for dist in min_dists:
		k_nearest = torch.cat((k_nearest, min_dict[dist]), dim = 0) # something is wrong with the concat
		ind = torch.nonzero((dataset == min_dict[dist]).sum(dim=1) == dataset.size(1))
		inds.append(ind)
	k_nearest = k_nearest[71:]
	k_nearest = k_nearest.view(-1, 71)
	return k_nearest, inds

def knn_all_sample_particles(sample_particles, X_knn, Y_knn, k):
	#  functionalize : the function takes in the whole dataset, k,
	#  and the sample particle to find the k nearest neighbors. It returns
	#  the k nearest neighbors in the real dataset, as well as the emmitances
	#  of those neighbors.
	# includes all of the dataset into X and Y
	#_, X_knn, Y_knn = load_dataset("/Users/MakenaLow/Downloads/CS236_Final_Project/Generative-Photocathode-Materials/unit_cell_data_16.csv", threshold=float("inf"))
	X_knn = torch.from_numpy(X_knn).type(torch.FloatTensor)
	knn_for_sample_particles = []
	inds_knn = []

	for one_sample_particle in sample_particles[1:3]:
		knn_for_sample_particle, inds = (knn(one_sample_particle, X_knn, k))
		knn_for_sample_particles.append(knn_for_sample_particle) # return value
		inds_knn.append(inds)

	lowest_emittance_neighbors = []
	lowest_emittance_of_neighbors = []
	for inds_knn_sample_particle in inds_knn:
		lowest_emittances = Y_knn[inds_knn_sample_particle] # can edit to take the min if wanted
		lowest_emittance, lowest_emittance_ind = np.amin(lowest_emittances), np.argmin(lowest_emittances)
		lowest_emittance_of_neighbors.append(lowest_emittance) # return value
		lowest_emittance_neighbors.append(torch.squeeze(X_knn[inds_knn_sample_particle[lowest_emittance_ind]]))

	return lowest_emittance_neighbors, lowest_emittance_of_neighbors
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
	elif loss_fn == "wasserstein_gp":
		error_real, _ = loss_wasserstein_gp(G, D, real_data, fake_data)

	error_real.backward()

	# Train on fake data
	pred_fake = D(fake_data)
	if loss_fn == "dflt":
		error_fake = loss(pred_fake, Variable(torch.ones(N, 1)))
	elif loss_fn == "nonsaturating":
		_, error_fake = nonsaturating_loss(G, D, real_data, fake_data)
	elif loss_fn == "wasserstein_gp":
		_, error_fake = loss_wasserstein_gp(G, D, real_data, fake_data)


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
	elif loss_fn == "wasserstein_gp":
		_, error = loss_wasserstein_gp(G, D, real_data, fake_data)

	error.backward()
	g_optimizer.step()
	return error

def train(X, num_batches, training_ratio = (1, 5), num_particle_samples=1000, G=None, D=None, set_args=None, train_cols=None, model_name='gpo-children'):
	if set_args:
		args = set_args
	loss_logger = Logger(model_name=model_name, data_name='loss')
	emit_logger = ELogger(model_name=model_name, data_name='emittance')

	loss = nn.BCELoss()  # Utilizing Binary Cross Entropy Loss
	if not G and not D:
		G, D, d_optimizer, g_optimizer = get_optimizers(args)
	else:
		_, _, d_optimizer, g_optimizer = get_optimizers(args)

	# Sample particles to examine progress
	test_noise = gen_noise(num_particle_samples, args.latent)

	for epoch in range(args.num_epochs):
		for n_batch, real_particle_batch in enumerate(X):
			real_particle_batch = torch.Tensor(real_particle_batch)
			N = real_particle_batch.shape[0] # This should be the batch size

			# Train the discriminator once
			real_data = Variable(real_particle_batch)  # gets a single row from the particles data
			d_noise = gen_noise(N, args.latent)  # generate a batch of noise vectors
			fake_data = G(d_noise).detach()


			# Training ratios
			discriminator_ratio, generator_ratio = training_ratio

			for i in range(discriminator_ratio):
				total_d_error, d_pred_real, d_pred_fake = train_discriminator(G, D, d_optimizer, loss, real_data, fake_data, args.loss_fn)

			# Train the generator 5 times for every 1 time we train the discriminator
			for d_step in range(generator_ratio):
				g_noise = gen_noise(N, args.latent)
				fake_data = G(g_noise)
				g_error = train_generator(G, D, g_optimizer, loss, real_data, fake_data, args.loss_fn)

			# Run logging to examine progress
			loss_logger.log(total_d_error, g_error, epoch, n_batch, num_batches)

		partial_eval_count = int(NUM_EVALUATORS)
		predictions = torch.zeros(num_particle_samples, partial_eval_count)
		sample_particle = G(test_noise)

		if train_cols:
			d = torch.zeros(num_particle_samples, 71)
			d[:,train_cols] = sample_particle
			sample_particle = d
		sample_particle = sample_particle.detach()

		for i in range(partial_eval_count):
			# Import the evaluator NN
			file = open('NN_evaluator_'+str(i)+'.sav', 'rb')
			clf = pk.load(file)
			# Using the evaluator NN, make a prediction on the generated particle
			predictions[:,i] = torch.tensor(clf.predict(sample_particle), dtype=torch.float32)
		if not train_cols and epoch == args.num_epochs-1:  # we do not want to log these since different noise sample
			mean_predicted_emittance = evaluate_generated_particles(G, num_particle_samples, args.latent)
		prediction = torch.mean(predictions)
		emittance_std = torch.std(predictions)

		emit_logger.log(epoch, prediction, emittance_std)
		# print(prediction, emittance_std)


	loss_logger.close()
	emit_logger.close()
	# return G, D, sample_particle, prediction
	return mean_predicted_emittance


def evaluate_generated_particles(G, num_particle_samples, latent):
	torch.set_printoptions(profile="full")

	# Generate NUM_PARTICLES test particles
	particles_emittances = []
	print("Testing particles...")
	particle_noise = gen_noise(num_particle_samples, latent)

	# Generate a test particle
	sample_particle = G(particle_noise)

	# Evaluator predicts on that particle
	sample_particle = sample_particle.detach()

	# Test the generated particle on all ten NN evaluators; then
	# take the average emittance prediction from the NN evaluators.
	predictions = torch.zeros(num_particle_samples, 10)

	for i in range(NUM_EVALUATORS):
		# Import the evaluator NN
		file = open('NN_evaluator_'+str(i)+'.sav', 'rb')
		clf = pk.load(file)
		# Using the evaluator NN, make a prediction on the generated particle
		predictions[:,i] = torch.tensor(clf.predict(sample_particle), dtype=torch.float32)

	# KNN analysis
	_, X_knn, Y_knn = load_dataset("../unit_cell_data_16.csv", threshold=float("inf"))
	lowest_emittance_neighbors, lowest_emittance_of_neighbors = knn_all_sample_particles(sample_particle, X_knn, Y_knn, 10)

	# print("Particles:")
	# print(sample_particle)
	# print("Emittances of Generated Particles:")
	# print(predictions)
	# print("Mean Emittance of Generated Particles Sample")
	# print(torch.mean(predictions))
	# print("Standard Deviation of Emittance of Generated Particles Sample")
	# print(torch.std(predictions))
	# print("Generated Parcticles Nearest Neighbors with Lowest Emmittance")
	# print(lowest_emittance_neighbors)
	# print("Emmittances of Nearest Neighbrors")
	# print(lowest_emittance_of_neighbors)
	torch.set_printoptions(profile="default")

	return torch.mean(predictions).item()


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

	_, X, Y = load_dataset("../unit_cell_data_16.csv", 0.2)
	X = batch_dataset(X, args.batch_size)
	num_batches = len(X)
	
	training_ratios = [(i, 1) for i in range(10, 1, -1)] + [(1, i) for i in range(1, 11)]
	print(training_ratios)
	average_emittance = [None for x in range(len(training_ratios))]

	for i, ratio in enumerate(training_ratios):
		print(str(i) + " of " + str(len(training_ratios)))
		emittances_curr_iter = []

		for j in range(1):

			mpe = train(X, num_batches, ratio, set_args=args, model_name=args.loss_fn)
			emittances_curr_iter.append(mpe)
		average_emittance[i] = torch.mean(torch.tensor(emittances_curr_iter)).item()
		print(average_emittance)

	with open("average_emittances.txt", "w+") as f:
		f.write(str(average_emittance))
		f.close()
