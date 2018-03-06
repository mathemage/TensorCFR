import tensorflow as tf

from assign_strategies_to_nodes import assign_strategies_to_nodes
from constants import PLAYER1
from domain.domain_01 import reach_probabilities_lvl0, node_to_IS_lvl0, IS_strategies_lvl0, node_to_IS_lvl1, \
	IS_strategies_lvl1, node_to_IS_lvl2, IS_strategies_lvl2, IS_acting_players_lvl0, IS_acting_players_lvl1, \
	IS_acting_players_lvl2
from utils.tensor_utils import print_tensors, expanded_multiply

# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

updating_player = PLAYER1

node_strategies_lvl0 = assign_strategies_to_nodes(IS_strategies_lvl0, node_to_IS_lvl0, name="node_strategies_lvl0")
node_strategies_lvl1 = assign_strategies_to_nodes(IS_strategies_lvl1, node_to_IS_lvl1, name="node_strategies_lvl1")
node_strategies_lvl2 = assign_strategies_to_nodes(IS_strategies_lvl2, node_to_IS_lvl2, name="node_strategies_lvl2")

# TODO generate node_cf_strategies_* with tf.where on node_strategies
node_cf_strategies_lvl0 = assign_strategies_to_nodes(IS_strategies_lvl0, node_to_IS_lvl0,
                                                     updating_player=updating_player,
                                                     acting_players=IS_acting_players_lvl0,
                                                     name="node_cf_strategies_lvl0")
node_cf_strategies_lvl1 = assign_strategies_to_nodes(IS_strategies_lvl1, node_to_IS_lvl1,
                                                     updating_player=updating_player,
                                                     acting_players=IS_acting_players_lvl1,
                                                     name="node_cf_strategies_lvl1")
node_cf_strategies_lvl2 = assign_strategies_to_nodes(IS_strategies_lvl2, node_to_IS_lvl2,
                                                     updating_player=updating_player,
                                                     acting_players=IS_acting_players_lvl2,
                                                     name="node_cf_strategies_lvl2")


def get_reach_probabilities():
	reach_probabilities_lvl1 = expanded_multiply(expandable_tensor=reach_probabilities_lvl0,
	                                             expanded_tensor=node_cf_strategies_lvl0, name="reach_probabilities_lvl1")
	reach_probabilities_lvl2 = expanded_multiply(expandable_tensor=reach_probabilities_lvl1,
	                                             expanded_tensor=node_cf_strategies_lvl1, name="reach_probabilities_lvl2")
	reach_probabilities_lvl3 = expanded_multiply(expandable_tensor=reach_probabilities_lvl2,
	                                             expanded_tensor=node_cf_strategies_lvl2, name="reach_probabilities_lvl3")
	return [reach_probabilities_lvl0, reach_probabilities_lvl1, reach_probabilities_lvl2, reach_probabilities_lvl3]


if __name__ == '__main__':
	reach_probabilities = get_reach_probabilities()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Level 0 ##########")
		print_tensors(sess, [reach_probabilities[0], node_to_IS_lvl0, IS_strategies_lvl0, node_strategies_lvl0,
												 node_cf_strategies_lvl0])
		print("########## Level 1 ##########")
		print_tensors(sess, [reach_probabilities[1], node_to_IS_lvl1, IS_strategies_lvl1, node_strategies_lvl1,
												 node_cf_strategies_lvl1])
		print("########## Level 2 ##########")
		print_tensors(sess, [reach_probabilities[2], node_to_IS_lvl2, IS_strategies_lvl2, node_strategies_lvl2,
												 node_cf_strategies_lvl2])
		print("########## Level 3 ##########")
		print_tensors(sess, [reach_probabilities[3]])
