#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import argparse
import logging
from abc import abstractmethod

import numpy as np

from src.commons.constants import SEED_FOR_TESTING
from src.utils.other_utils import get_current_timestamp


class AbstractNNRunner:
	def __init__(self, fixed_randomness=False):
		self.argparser = argparse.ArgumentParser()
		self.fixed_randomness = fixed_randomness
		self.args = None
		self.network = None
		self.epoch = None
		self.ckpt_basenames = []

		# NN preparation
		np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
		if self.fixed_randomness:
			print("Abstract: self.fixed_randomness is {}".format(self.fixed_randomness))
			np.random.seed(SEED_FOR_TESTING)  # Fix random seed
		self.parse_arguments()

	@property
	def default_extractor_arch(self):
		raise NotImplementedError

	@property
	def default_regressor_arch(self):
		raise NotImplementedError

	def add_arguments_to_argparser(self):
		raise NotImplementedError

	def parse_arguments(self):
		self.add_arguments_to_argparser()
		self.args = self.argparser.parse_args()
		print("args: {}".format(self.args))

	def create_logdir(self):
		import datetime
		import re
		import os

		del self.args.dataset_directory
		del self.args.ckpt_dir
		del self.args.ckpt_basename
		self.args.logdir = "logs/{}-{}-{}".format(
			os.path.basename(self.__class__.__name__),
			datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
			",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
			          for key, value in sorted(vars(self.args).items())))
		)
		if not os.path.exists("logs"):
			os.mkdir("logs")  # TF 1.6 will do this by itself

	@abstractmethod
	def init_datasets(self, dataset_directory):
		pass

	@abstractmethod
	def construct_network(self):
		pass

	def train_one_epoch(self, trainset):
		while not trainset.epoch_finished():
			logging.info("[epoch #{}, batch #{}] Training...".format(self.epoch, trainset.batch_id))
			reaches, targets = trainset.next_batch(self.args.batch_size)
			self.network.train(reaches, targets)

	def evaluate_devset(self, devset):
		devset_error_mse, devset_error_infinity = self.network.evaluate("dev", devset.features, devset.targets)
		logging.info("[epoch #{}] dev MSE {}, \tdev L-infinity error {}".format(
			self.epoch,
			devset_error_mse, devset_error_infinity)
		)

	def evaluate_testset(self, testset):
		testset_error_mse, testset_error_infinity = self.network.evaluate("test", testset.features, testset.targets)
		# logging.info("\nmean squared error on testset: {}".format(testset_error_mse))   # TODO
		# logging.info("L-infinity error on testset: {}".format(testset_error_infinity))  # TODO
		print("\nmean squared error on testset: {}".format(testset_error_mse))
		print("L-infinity error on testset: {}".format(testset_error_infinity))

	def showcase_predictions(self, trainset):
		print("\nPredictions of initial 2 training examples:")
		print(self.network.predict(trainset.features[:2]))

	def run_neural_net(self, ckpt_every=None, ckpt_dir=None):
		dataset_directory = self.args.dataset_directory
		self.create_logdir()

		devset, testset, trainset = self.init_datasets(dataset_directory)
		self.network = self.construct_network()

		if ckpt_dir is None:
			ckpt_dir = self.args.logdir

		for self.epoch in range(self.args.epochs):
			self.train_one_epoch(trainset)
			self.evaluate_devset(devset)

			# checkpoint every `ckpt_every` epochs
			if ckpt_every and ckpt_dir is not None and int(self.epoch) % int(ckpt_every) == 0:
				ckpt_basename = "epoch_{}_{}".format(str(self.epoch), str(get_current_timestamp()))
				self.ckpt_basenames.append(ckpt_basename)
				self.network.save_to_ckpt(ckpt_dir, ckpt_basename)

		ckpt_basename = "final_{}.ckpt".format(get_current_timestamp())
		self.ckpt_basenames.append(ckpt_basename)
		self.network.save_to_ckpt(ckpt_dir, ckpt_basename)  # final model

		self.evaluate_testset(testset)
		self.showcase_predictions(trainset)

	def restore_from_ckpt(self, ckpt_dir=None, ckpt_basename=None):
		if (ckpt_dir is None) or (ckpt_basename is None):
			ckpt_dir = self.args.ckpt_dir
			ckpt_basename = self.args.ckpt_basename
		self.create_logdir()
		self.network = self.construct_network()
		self.network.restore_from_ckpt(ckpt_dir, ckpt_basename)

	def run_neural_net_from_ckpt(self, ckpt_dir=None, ckpt_basename=None):
		dataset_directory = self.args.dataset_directory
		_, testset, trainset = self.init_datasets(dataset_directory)

		self.restore_from_ckpt(ckpt_dir, ckpt_basename)
		self.evaluate_testset(testset)
		self.showcase_predictions(trainset)
