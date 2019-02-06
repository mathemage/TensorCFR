#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_AVERAGING_DELAY
from src.domains.FlattenedDomain import FlattenedDomain
#from src.nn.NNMockUp import NNMockUp
from src.utils.tf_utils import get_default_config_proto, print_tensor, masked_assign
from src.nn.data.postprocessing_ranges import tensorcfr_to_nn_input,load_nn,stack_public_state_predictions,nn_out_to_tensorcfr_in
from src.nn.data.preprocessing_ranges import load_input_mask,load_output_mask,load_history_identifier,load_infoset_list,load_infoset_hist_ids,filter_by_card_combination,filter_by_public_state
import numpy as np

class TensorCFR_Goofstack(TensorCFRFixedTrunkStrategies):
	def __init__(self, domain: FlattenedDomain, neural_net=None, trunk_depth=0):
		"""
		Constructor for an instance of TensorCFR_NN algorithm with given parameters (as a TensorFlow computation graph).

		:param domain: The domain of the game (as an instance of class `FlattenedDomain`). TensorCFR (the CFR+ algorithm
		with neural network for prediction of nodal expected values) will be launched for this game.
	  :param trunk_depth: The number of levels of the trunk where the strategies are kept fixed. It should be an integer between `0` to
		`self.domain.levels`. It defaults to `0` (no trunk).
		"""
		super().__init__(
			domain,
			trunk_depth,
			levels=trunk_depth + 1,  # `trunk_depth` levels where strategy is fixed + the final one at the bottom
			acting_depth=trunk_depth
		)

		self.neural_net = load_nn()
		self.session = tf.Session(config=get_default_config_proto())
		self.construct_computation_graph()
		self.mask = load_input_mask()
		self.output_mask = load_output_mask()
		self.hist_id = load_history_identifier()
		self.infoset_list = load_infoset_list()
		self.infoset_hist_ids = load_infoset_hist_ids().iloc[:, :120]
		self.public_states_list = [(x, y, z) for x in [0, 1, -1] for y in [0, 1, -1] for z in [0, 1, -1]]
		self.tensor_cfr_in_mask = np.zeros(120 ** 2)
		with tf.variable_scope("initialization"):
			setup_messages, feed_dictionary = self.set_up_feed_dictionary(method="by-domain")
			print(setup_messages)
		self.average_strategies_over_steps = None
		self.session.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)

	def construct_lowest_expected_values(self, player_name, signum):
		with tf.variable_scope("level{}".format(self.levels - 1)):
			lowest_utilities = self.domain.utilities[self.levels - 1]
			self.predicted_equilibrial_values = tf.placeholder_with_default(
				lowest_utilities,
				shape=lowest_utilities.shape,
				name="predicted_equilibrial_values"
			)
			self.expected_values[self.levels - 1] = tf.multiply(
				signum,
				self.predicted_equilibrial_values,
				name="expected_values_lvl{}_for_{}".format(self.levels - 1, player_name)
			)

	def update_strategy_of_updating_player(self, acting_player=None):  # override not to fix trunk
		"""
		Update for the strategy for the given `acting_player`.

		Args:
			:param acting_player: A variable. An index of the player whose strategies are to be updated.

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		if acting_player is None:
			acting_player = self.domain.current_updating_player
		infoset_strategies_matched_to_regrets = self.get_strategy_matched_to_regrets()
		infoset_acting_players = self.domain.get_infoset_acting_players()
		ops_update_infoset_strategies = [None] * self.acting_depth
		with tf.variable_scope("update_strategy_of_updating_player"):
			for level in range(self.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					infosets_of_acting_player = tf.reshape(
						# `tf.reshape` to force "shape of 2D tensor" == [number of infosets, 1]
						tf.equal(infoset_acting_players[level], acting_player),
						shape=[self.domain.current_infoset_strategies[level].shape[0]],
						name="infosets_of_updating_player_lvl{}".format(level)
					)
					ops_update_infoset_strategies[level] = masked_assign(
						ref=self.domain.current_infoset_strategies[level],
						mask=infosets_of_acting_player,
						value=infoset_strategies_matched_to_regrets[level],
						name="op_update_infoset_strategies_lvl{}".format(level)
					)
			return ops_update_infoset_strategies

	def construct_computation_graph(self):
		self.cfr_step_op = self.do_cfr_step()
		self.input_ranges = self.get_nodal_range_probabilities()

	def tensorcfr_to_nn_input(self,tensor_cfr_out=None):

		##TODO get ranges from tensorcfrfixestrunk. bring them into format [public_state,ranges p1] for each publicstate
		## TODO implement range of ifnoset in tensorcfr. its easier

		mask = self.mask.copy()
		hist_id = self.hist_id.copy()

		for public_state in self.public_states_list:

			df_by_public_state = filter_by_public_state(hist_id, public_state)

			for cards in mask.columns[3:123]:
				## for player 1

				cards_df = filter_by_card_combination(df_by_public_state, cards, 1)

				if cards_df.shape[0] >= 1:

					# puts range of p1 in of infoset "cards" of public state "public_state" into mask

					mask.loc["".join(tuple(map(str, public_state))), cards] = float(
						tensor_cfr_out.iloc[tensor_cfr_out.index == cards_df.index[0], 0])

				else:

					mask.loc["".join(tuple(map(str, public_state))), cards] = 0

			for cards in mask.columns[123:]:

				cards_df = filter_by_card_combination(df_by_public_state, cards, 2)

				if cards_df.shape[0] == 1:

					mask.loc["".join(tuple(map(str, public_state))), cards] = float(
						tensor_cfr_out.iloc[tensor_cfr_out.index == cards_df.index[0], 1])

		permuted_input = tf.expand_dims(
			permutate_op.forward(input_reaches),  # permute input reach probabilities
			axis=0,  # simulate batch size of 1 for prediction
			name="expanded_permuted_input"
		)

		np_permuted_input = self.session.run(permuted_input)

				elif cards_df.shape[0] > 1:

					mask.loc["".join(tuple(map(str, public_state))), cards] = float(
						tensor_cfr_out.iloc[tensor_cfr_out.index == cards_df.index[0], 1])


				elif cards_df.shape[0] == 0:

					mask.loc["".join(tuple(map(str, public_state))), cards] = 0

		return mask

	def nn_out_to_tensorcfr_in(self,nn_out=None):

		if nn_out.shape != (27, 120):
			raise ValueError

		else:
			## this version is only for nns that output cfv of p1. meaning a vector of size 120 for each public state

			tensor_cfr_in = self.tensor_cfr_in_mask.copy()

			for id in self.infoset_list:

				tensor_cfr_in[id] = nn_out.iloc[np.where(self.infoset_hist_ids == id)]

			return tensor_cfr_in

	def predict_equilibrial_values(self, input_ranges=None, name="predictions"):
		if input_ranges is None:
			input_ranges = self.input_ranges

		tensorcfr_in = tensorcfr_to_nn_input(input_ranges)

		nn_out = np.vstack([self.neural_net.predict(tensorcfr_in[i,:]) for i in range(tensorcfr_in.shape[0])])

		predicted_equilibrial_values = nn_out_to_tensorcfr_in(nn_out)

		# permute back the expected values
		 # TODO make into numpy array
		return tf.identity(predicted_equilibrial_values, name=name)

	def run_cfr(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY, verbose=False,
	            register_strategies_on_step=None):
		if register_strategies_on_step is None:
			register_strategies_on_step = [total_steps - 1]  # by default, register just the last iteration
		self.average_strategies_over_steps = dict()        # reset the dict

		self.cfr_parameters = {
			"total_steps"    : total_steps,
			"averaging_delay": delay,
			"trunk_depth"    : self.trunk_depth,
		}
		self.set_up_cfr_parameters(delay, total_steps)
		self.set_log_directory()
		with tf.summary.FileWriter(self.log_directory, tf.get_default_graph()):
			for step in range(total_steps):
				print("\n########## CFR step {} ##########".format(step))
				predicted_equilibrial_values = self.predict_equilibrial_values()
				if verbose:
					print("Before:")
					print_tensor(self.session, self.input_reaches)
					print_tensor(self.session, predicted_equilibrial_values)
				np_predicted_equilibrial_values = self.session.run(predicted_equilibrial_values)
				self.session.run(self.cfr_step_op, {self.predicted_equilibrial_values: np_predicted_equilibrial_values})
				if verbose:
					print("After:")
					print_tensor(self.session, self.input_reaches)

				if step in register_strategies_on_step:
					self.average_strategies_over_steps["average_strategy_step{}".format(step)] = [
						self.session.run(strategy).tolist() for strategy in self.average_infoset_strategies[:self.trunk_depth]
					]
