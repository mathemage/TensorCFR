#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_AVERAGING_DELAY
from src.domains.FlattenedDomain import FlattenedDomain
from src.nn.NNMockUp import NNMockUp
from src.utils.tf_utils import get_default_config_proto, print_tensor, masked_assign


def get_sorted_permutation():
	return [2, 1, 0]


class TensorCFR_NN(TensorCFRFixedTrunkStrategies):
	def __init__(self, domain: FlattenedDomain, neural_net=None, nn_input_permutation=None, trunk_depth=0):
		"""
		Constructor for an instance of TensorCFR algorithm with given parameters (as a TensorFlow computation graph).

		:param domain: The domain of the game (as an instance of class `FlattenedDomain`). TensorCFR (the CFR+ algorithm)
		 will be launched for this game.
		:param trunk_depth: The number of levels of the trunk where the strategies are kept fixed. It should be an integer
			 between `0` to `self.domain.levels`. It defaults to `0` (no trunk).
		"""
		super().__init__(domain, trunk_depth)
		self.neural_net = neural_net if neural_net is not None else NNMockUp()
		self.nn_input_permutation = nn_input_permutation if nn_input_permutation is not None else get_sorted_permutation()
		self.session = tf.Session(config=get_default_config_proto())
		self.construct_computation_graph()
		with tf.variable_scope("initialization"):
			setup_messages, feed_dictionary = self.set_up_feed_dictionary(method="by-domain")
			print(setup_messages)
		self.session.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)

	# TODO modify topdown
	# TODO modify bottomup

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
		ops_update_infoset_strategies = [None] * self.domain.acting_depth
		with tf.variable_scope("update_strategy_of_updating_player"):
			for level in range(self.domain.acting_depth):       # TODO update only at trunk
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

	def cfr_strategies_after_fixed_trunk(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY,
	                                     storing_strategies=False, verbose=False, register_strategies_on_step=list()):
		# a list of returned average strategies
		# the parameter `register_strategies_on_step` is used to determine which strategy export
		return_average_strategies = list()

		# if the `register_strategies_on_step` list is empty, register just the last iteration
		if len(register_strategies_on_step) == 0:
			register_strategies_on_step.append(total_steps - 1)

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
				self.session.run(self.cfr_step_op)
				if step in register_strategies_on_step:
					return_average_strategies.append({
						"step": step,
						"average_strategy": [self.session.run(x) for x in self.average_infoset_strategies]
					})

				if verbose:
					print_tensor(self.session, self.get_nodal_reaches_at_trunk_depth())
					print_tensor(self.session, self.predict_equilibrial_values())
		return return_average_strategies

	def predict_equilibrial_values(self, input_reaches=None, name="permuted_predictions"):
		if input_reaches is None:
			input_reaches = self.get_nodal_reaches_at_trunk_depth()
		permutate_op = tf.contrib.distributions.bijectors.Permute(permutation=self.nn_input_permutation)

		permuted_input = tf.expand_dims(
			permutate_op.forward(input_reaches),    # permute input reach probabilities
			axis=0,                                 # simulate batch size of 1 for prediction
			name="expanded_permuted_input"
		)

		np_permuted_input = self.session.run(permuted_input)

		# use neural net to predict equilibrium values
		predicted_equilibrial_values = self.neural_net.predict(np_permuted_input)

		# permute back the expected values
		permuted_predictions = permutate_op.inverse(predicted_equilibrial_values)
		return tf.identity(permuted_predictions, name=name)