#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import numpy as np
import tensorflow as tf

from src.commons.constants import SEED_FOR_TESTING, FLOAT_DTYPE
from src.nn.ConvNet_IIGS6Lvl10 import ConvNet_IIGS6Lvl10
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ
from src.utils.other_utils import activate_script

FIXED_RANDOMNESS = False


class ConvNetBaseline_IIGS6Lvl10(ConvNet_IIGS6Lvl10):
	def construct_input(self):
		with tf.variable_scope("input"):
			self.input_reaches = tf.placeholder(
				FLOAT_DTYPE,
				[None, self.NUM_NODES],
				name="input_reaches"
			)
			self._one_hot_features_tf = tf.constant(
				self._one_hot_features_np,
				dtype=FLOAT_DTYPE,
				name="one_hot_features"
			)
			print("one_hot_features.shape: {}".format(self._one_hot_features_tf.shape))
			self.tiled_features = tf.tile(
				tf.expand_dims(self._one_hot_features_tf, axis=0),
				multiples=[tf.shape(self.input_reaches)[0], 1, 1],
				name="tiled_1hot_features"
			)
			self.latest_layer = tf.transpose(  # channels first for GPU computation
				self.tiled_features,
				perm=[0, 2, 1],
				name="input_channels_first_NCL"  # [batch, channels, lengths] == [batch_size, INPUT_FEATURES_DIM, NUM_NODES]
			)
		print("Input constructed")
		self.targets = tf.placeholder(FLOAT_DTYPE, [None, self.NUM_NODES], name="targets")
		print("Targets constructed")
		self.print_operations_count()

	def construct_summaries(self, args):
		with tf.variable_scope("summaries"):
			self.summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
		self.summaries = {}
		with self.summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(args.batch_size):
			self.summaries["train"] = [
				tf.contrib.summary.scalar("train/loss", self.loss),
				tf.contrib.summary.scalar("train/mean_squared_error", self.mean_squared_error),
				tf.contrib.summary.scalar("train/l_infinity_error", self.l_infinity_error)
			]
		print("Summaries[train] constructed")
		self.print_operations_count()
		with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
			for dataset in ["dev", "test"]:
				self.summaries[dataset] = [
					tf.contrib.summary.scalar(dataset + "/loss", self.loss),
					tf.contrib.summary.scalar(dataset + "/mean_squared_error", self.mean_squared_error),
					tf.contrib.summary.scalar(dataset + "/l_infinity_error", self.l_infinity_error)
				]
		print("Summaries[dev/test] constructed")
		self.print_operations_count()


if __name__ == '__main__' and activate_script():
	import argparse
	import datetime
	import os
	import re

	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	if FIXED_RANDOMNESS:
		np.random.seed(SEED_FOR_TESTING)  # Fix random seed

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
	parser.add_argument("--dataset_directory", default="data/IIGS6Lvl10/minimal_dataset/2",
	                    help="Relative path to dataset folder.")
	parser.add_argument("--extractor", default="C-{}".format(ConvNetBaseline_IIGS6Lvl10.INPUT_FEATURES_DIM), type=str,
	                    help="Description of the feature extactor architecture.")
	parser.add_argument("--regressor", default="C-{}".format(ConvNetBaseline_IIGS6Lvl10.INPUT_FEATURES_DIM), type=str,
	                    help="Description of the value regressor architecture.")
	parser.add_argument("--epochs", default=50000, type=int, help="Number of epochs.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

	args = parser.parse_args()
	print("args: {}".format(args))

	# Create logdir name
	dataset_directory = args.dataset_directory
	del args.dataset_directory
	args.logdir = "logs/{}-{}-{}".format(
		os.path.basename(__file__),
		datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
		",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
	)
	if not os.path.exists("logs"):
		os.mkdir("logs")  # TF 1.6 will do this by itself

	# Load the data
	script_directory = os.path.dirname(os.path.abspath(__file__))
	npz_basename = "IIGS6_1_6_false_true_lvl10"
	trainset = DatasetFromNPZ("{}/{}/{}_train.npz".format(script_directory, dataset_directory, npz_basename))
	devset = DatasetFromNPZ("{}/{}/{}_dev.npz".format(script_directory, dataset_directory, npz_basename))
	testset = DatasetFromNPZ("{}/{}/{}_test.npz".format(script_directory, dataset_directory, npz_basename))

	# Construct the network
	network = ConvNetBaseline_IIGS6Lvl10(threads=args.threads)
	network.construct(args)

	# Train
	for epoch in range(args.epochs):
		while not trainset.epoch_finished():
			reaches, targets = trainset.next_batch(args.batch_size)
			network.train(reaches, targets)

		# Evaluate on development set
		devset_error_mse, devset_error_infinity = network.evaluate("dev", devset.features, devset.targets)
		print("[epoch #{}] dev MSE {}, \tdev L-infinity error {}".format(epoch, devset_error_mse, devset_error_infinity))

	# Evaluate on test set
	testset_error_mse, testset_error_infinity = network.evaluate("test", testset.features, testset.targets)
	print()
	print("mean squared error on testset: {}".format(testset_error_mse))
	print("L-infinity error on testset: {}".format(testset_error_infinity))

	print()
	print("Predictions of initial 2 training examples:")
	print(network.predict(trainset.features[:2]))
