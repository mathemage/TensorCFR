#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py

from src.nn.AbstractNNRunner import AbstractNNRunner
from src.nn.ConvNet_IIGS6Lvl10 import ConvNet_IIGS6Lvl10
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ

FIXED_RANDOMNESS = False


class Runner_CNN_IIGS6Lvl10_NPZ(AbstractNNRunner):
	@staticmethod
	def datasets_from_npz(dataset_directory, script_directory):
		npz_basename = "IIGS6_1_6_false_true_lvl10"
		trainset = DatasetFromNPZ("{}/{}/{}_train.npz".format(script_directory, dataset_directory, npz_basename))
		devset = DatasetFromNPZ("{}/{}/{}_dev.npz".format(script_directory, dataset_directory, npz_basename))
		testset = DatasetFromNPZ("{}/{}/{}_test.npz".format(script_directory, dataset_directory, npz_basename))
		return devset, testset, trainset

	@staticmethod
	def init_datasets(dataset_directory):
		import os
		script_directory = os.path.dirname(os.path.abspath(__file__))
		devset, testset, trainset = Runner_CNN_IIGS6Lvl10_NPZ.datasets_from_npz(dataset_directory, script_directory)
		return devset, testset, trainset

	@staticmethod
	def construct_network(args):
		network = ConvNet_IIGS6Lvl10(threads=args.threads, fixed_randomness=FIXED_RANDOMNESS)
		network.construct(args)
		return network


# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = True


if __name__ == '__main__' and ACTIVATE_FILE:
	Runner_CNN_IIGS6Lvl10_NPZ.run_neural_net()
