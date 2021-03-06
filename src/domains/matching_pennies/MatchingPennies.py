#!/usr/bin/env python3

import tensorflow as tf

import src.domains.matching_pennies.matching_pennies_as_numpy_values as mp
from src.domains.Domain import Domain


def get_domain_matching_pennies():
	return Domain(
		domain_name="matching_pennies",
		domain_parameters=[],
		actions_per_levels=mp.actions_per_levels,
		node_to_infoset=mp.node_to_infoset,
		node_types=mp.node_types,
		utilities=mp.utilities,
		infoset_acting_players=mp.infoset_acting_players,
		initial_infoset_strategies=mp.initial_infoset_strategies,
	)


if __name__ == '__main__':
	matching_pennies = get_domain_matching_pennies()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		matching_pennies.print_domain(sess)
