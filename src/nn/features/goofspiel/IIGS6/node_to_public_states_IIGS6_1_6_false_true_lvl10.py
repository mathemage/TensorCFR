#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.nn.features.goofspiel.IIGS6.game_constants import FEATURES_BASENAME, NAMES_OF_FEATURE_CSV
from src.utils.other_utils import get_features_dataframe


def get_node_to_public_state(verbose=False):
	script_directory = os.path.dirname(os.path.abspath(__file__))
	csv_basename = FEATURES_BASENAME
	csv_filename = os.path.join(script_directory, "{}.csv".format(csv_basename))
	features = get_features_dataframe(csv_filename, names=NAMES_OF_FEATURE_CSV)

	grouped_features = features.groupby(['round1', 'round2', 'round3'])
	df_public_states_sizes = DataFrame(
		{"public_state_size": (grouped_features.size())}
	).reset_index()
	np_public_states_sizes = df_public_states_sizes["public_state_size"].values
	node_to_public_state = np.array([
		i
		for i, size in enumerate(np_public_states_sizes)
		for _ in range(size)
	])

	if verbose:
		print("grouped_features:\n{}.".format(grouped_features.head()))
		print("df_public_states_sizes:\n{}\n".format(df_public_states_sizes))
		print("np_public_states_sizes:\n{}\n".format(np_public_states_sizes))
		print("node_to_public_state:\n{}\n".format(node_to_public_state))

	return node_to_public_state


if __name__ == '__main__':
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	node_to_public_state_mapping = get_node_to_public_state()
	print("node_to_public_state_mapping:\n{}\n".format(node_to_public_state_mapping))