#!/usr/bin/env python3

import tensorflow as tf

from constants import NON_TERMINAL_UTILITY, INNER_NODE, TERMINAL_NODE, IMAGINARY_NODE, CHANCE_PLAYER, PLAYER1, \
	PLAYER2, NO_ACTING_PLAYER
from utils.tensor_utils import print_tensors, masked_assign

# custom-made game: see doc/domain_01_via_drawing.png and doc/domain_01_via_gambit.png

actions_per_levels = [5, 3, 2]  # maximum number of actions per each level (0, 1, 2)
levels = len(actions_per_levels) + 1  # accounting for 0th level

# Node to IS 
node_to_IS = list()
node_to_IS.append( tf.Variable( 0, name="node_to_IS_lvl0" ) )
node_to_IS.append( tf.Variable( [0, 1, 2, 2, 3], name="node_to_IS_lvl1" ) )
node_to_IS.append( tf.Variable( [[0, 1, 7],  # s5, s6, s7
                                 [2, 2, 8],  # s8, s9, s10
                                 [3, 4, 8],  # s11, s12, s13
                                 [3, 4, 8],  # s14, s15, s16
                                 [7, 5, 6]], # s17, s18, s19
                                 name="node_to_IS_lvl2") )

# Reach probabilities
reach_probabilities = list()
reach_probabilities.append( tf.Variable(1.0, name="reach_probabilities_lvl0") )

# Shape 
shape = list()
shape.append( actions_per_levels[:0] )
shape.append( actions_per_levels[:1] )
shape.append( actions_per_levels[:2] )
shape.append( actions_per_levels[:3] )

# Node types
node_types = list()
node_types.append( tf.Variable(INNER_NODE, name="node_types_lvl0") )
node_types.append( tf.Variable([INNER_NODE] * 5, name="node_types_lvl1") )
node_types.append( tf.Variable([[INNER_NODE, INNER_NODE, TERMINAL_NODE],   # s5, s6, s7
                                [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s8, s9, s10
                                [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s11, s12, s13
                                [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s14, s15, s16
                                [TERMINAL_NODE, INNER_NODE, INNER_NODE]],  # s17, s18, s19
                                name="node_types_lvl2") )
node_types.append( tf.Variable( tf.fill( value=TERMINAL_NODE, dims=shape[3] ), name="node_types_lvl3" ) )

# Utilities
utilities = list()
utilities.append( tf.fill( value=NON_TERMINAL_UTILITY, dims=shape[0], name="utilities_lvl0" ) )
utilities.append( tf.fill( value=NON_TERMINAL_UTILITY, dims=shape[1], name="utilities_lvl1" ) )
utilities.append( tf.Variable( tf.fill( value=NON_TERMINAL_UTILITY, dims=shape[2] ) ) )
utilities.append( tf.Variable( tf.fill( value=NON_TERMINAL_UTILITY, dims=shape[3] ) ) )

# IS acting player
IS_acting_players = list()
IS_acting_players.append( tf.Variable(CHANCE_PLAYER, name="IS_acting_players_lvl0") ) 
IS_acting_players.append( tf.Variable( [PLAYER1,         # I1,0
                                        PLAYER2,         # I1,1
                                        PLAYER2,         # I1,2
                                        CHANCE_PLAYER],  # I1,3
                                        name="IS_acting_players_lvl1") )
IS_acting_players.append( tf.Variable( [PLAYER1,            # of I2,0
                                        PLAYER2,            # of I2,1
                                        PLAYER1,            # of I2,2
                                        PLAYER2,            # of I2,3
                                        CHANCE_PLAYER,      # of I2,4
                                        PLAYER1,            # of I2,5
                                        PLAYER2,            # of I2,6
                                        NO_ACTING_PLAYER,   # of I2,7 ... pseudo-infoset of terminal nodes
                                        NO_ACTING_PLAYER],  # of I2,8 ... pseudo-infoset of imaginary nodes
                                        name="IS_acting_players_lvl2") )


########## Level 0 ##########
# I0,0 = { s } ... root node, the chance player acts here
# there are 5 actions in node s

reach_probabilities_lvl0 = reach_probabilities[0]

#shape_lvl0 = actions_per_levels[:0]
shape_lvl0 = shape[0]

node_to_IS_lvl0 = node_to_IS[0]
#node_to_IS_lvl0 = tf.Variable(0, name="node_to_IS_lvl0")  # NOTE: the value above is not [1] in order to remove 1
																													# redundant '[]' represented by choice of empty sequence {}

#node_types_lvl0 = tf.Variable(INNER_NODE, name="node_types_lvl0")
node_types_lvl0 = node_types[0]

#utilities_lvl0 = tf.fill(value=NON_TERMINAL_UTILITY, dims=shape_lvl0, name="utilities_lvl0")
utilities_lvl0 = utilities[0]

IS_acting_players_lvl0 = IS_acting_players[0]

IS_strategies_lvl0 = tf.Variable([[0.5, .25, 0.1, 0.1, .05]],  # of I0,0
                                 name="IS_strategies_lvl0")

########## Level 1 ##########
# I1,0 = { s' }
# I1,1 = { s1 }
# I1,2 = { s2, s3 }
# I1,3 = { s4 } ... chance node
# each node has 3 actions

#shape_lvl1 = actions_per_levels[:1]
shape_lvl1 = shape[1]

node_to_IS_lvl1 = node_to_IS[1]
#node_to_IS_lvl1 = tf.Variable([0, 1, 2, 2, 3], name="node_to_IS_lvl1")

#node_types_lvl1 = tf.Variable([INNER_NODE] * 5, name="node_types_lvl1")
node_types_lvl1 = node_types[1]

#utilities_lvl1 = tf.fill(value=NON_TERMINAL_UTILITY, dims=shape_lvl1, name="utilities_lvl1")
utilities_lvl1 = utilities[1]

IS_acting_players_lvl1 = IS_acting_players[1]

IS_strategies_lvl1 = tf.Variable([[0.5, 0.4, 0.1],   # of I1,0
	                                [0.1, 0.9, 0.0],   # of I1,1
                                  [0.2, 0.8, 0.0],   # of I1,2
                                  [0.3, 0.3, 0.3]],  # of I1,3
                                 name="IS_strategies_lvl1")

########## Level 2 ##########
# I2,0 = { s5 }
# I2,1 = { s6 }
# I2,2 = { s8, s9 }
# I2,3 = { s11, s14 }
# I2,4 = { s12, s15 }
# I2,5 = { s18 }
# I2,6 = { s19 }
# I2,7 = { s7, s17 } ... terminal nodes
# I2,8 = { s10, s13, s16 } ... imaginary nodes
# each node has 2 actions

#shape_lvl2 = actions_per_levels[:2]
shape_lvl2 = shape[2]

node_to_IS_lvl2 = node_to_IS[2]
"""
node_to_IS_lvl2 = tf.Variable([[0, 1, 7],  # s5, s6, s7
                               [2, 2, 8],  # s8, s9, s10
                               [3, 4, 8],  # s11, s12, s13
                               [3, 4, 8],  # s14, s15, s16
                               [7, 5, 6]],  # s17, s18, s19
                              name="node_to_IS_lvl2")
"""

"""
node_types_lvl2 = tf.Variable([[INNER_NODE, INNER_NODE, TERMINAL_NODE],   # s5, s6, s7
                               [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s8, s9, s10
                               [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s11, s12, s13
                               [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s14, s15, s16
                               [TERMINAL_NODE, INNER_NODE, INNER_NODE]],  # s17, s18, s19
                              name="node_types_lvl2")
"""
node_types_lvl2 = node_types[2]


#utilities_lvl2 = tf.Variable(tf.fill(value=NON_TERMINAL_UTILITY, dims=shape_lvl2))
utilities_lvl2 = utilities[2]

mask_terminals_lvl2 = tf.equal(node_types_lvl2, TERMINAL_NODE)

terminal_values_lvl2 = tf.reshape(tf.range(10, 160, delta=10.0), shape_lvl2)

utilities_lvl2 = masked_assign(ref=utilities_lvl2, value=terminal_values_lvl2, mask=mask_terminals_lvl2,
                               name="utilities_lvl2")

IS_acting_players_lvl2 = IS_acting_players[2]

IS_strategies_lvl2 = tf.Variable([[0.15, 0.85],   # of I2,0
                                  [0.70, 0.30],   # of I2,1
                                  [0.25, 0.75],   # of I2,2
                                  [0.50, 0.50],   # of I2,3
                                  [0.10, 0.90],   # of I2,4
                                  [0.45, 0.55],   # of I2,5
                                  [0.40, 0.60],   # of I2,6
                                  [0.00, 0.00],   # of I2,7 ... terminal nodes <- mock-up zero strategy
                                  [0.00, 0.00]],  # of I2,8 ... imaginary nodes <- mock-up zero strategy
                                 name="IS_strategies_lvl2")

########## Level 3 ##########
# There are never any infosets in the final layer, only terminal / imaginary nodes.

#shape_lvl3 = actions_per_levels[:3]
shape_lvl3 = shape[3]

node_types_lvl3 = node_types[3]

indices_imaginary_nodes_lvl3 = tf.constant([[0, 2],   # children of s7
                                            [1, 2],   # children of s10
                                            [2, 2],   # children of s13
                                            [3, 2],   # children of s16
                                            [4, 0]],  # children of s17
                                           name="indices_imaginary_nodes_lvl3")

node_types_lvl3 = tf.scatter_nd_update(ref=node_types_lvl3, indices=indices_imaginary_nodes_lvl3,
                                       updates=tf.fill(value=IMAGINARY_NODE, dims=indices_imaginary_nodes_lvl3.shape),
                                       name="node_types_lvl3")

#utilities_lvl3 = tf.Variable(tf.fill(value=NON_TERMINAL_UTILITY, dims=shape_lvl3))
utilities_lvl3 = utilities[3]

mask_terminals_lvl3 = tf.equal(node_types_lvl3, TERMINAL_NODE)

terminal_values_lvl3 = tf.reshape(tf.range(10, 310, delta=10.0), shape_lvl3)

utilities_lvl3 = masked_assign(ref=utilities_lvl3, value=terminal_values_lvl3, mask=mask_terminals_lvl3,
                               name="utilities_lvl3")

########## miscellaneous tensors ##########
cf_values_IS_actions = [tf.Variable(tf.zeros_like(IS_strategies_lvl0), name="cf_values_IS_actions_lvl0"),
                        tf.Variable(tf.zeros_like(IS_strategies_lvl1), name="cf_values_IS_actions_lvl1"),
                        tf.Variable(tf.zeros_like(IS_strategies_lvl2), name="cf_values_IS_actions_lvl2")]
positive_cumulative_regrets = [tf.Variable(tf.zeros_like(IS_strategies_lvl0), name="pos_cumul_regrets_lvl0"),
                               tf.Variable(tf.zeros_like(IS_strategies_lvl1), name="pos_cumul_regrets_lvl1"),
                               tf.Variable(tf.zeros_like(IS_strategies_lvl2), name="pos_cumul_regrets_lvl2")]


def get_node_types():
	node_types = [node_types_lvl0, node_types_lvl1, node_types_lvl2, node_types_lvl3]
	return node_types


def get_node_to_IS():
	node_to_IS = [node_to_IS_lvl0, node_to_IS_lvl1, node_to_IS_lvl2]
	return node_to_IS


# TODO use get_IS_strategies elsewhere in the code
def get_IS_strategies():
	IS_strategies = [IS_strategies_lvl0, IS_strategies_lvl1, IS_strategies_lvl2]
	return IS_strategies


if __name__ == '__main__':
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Level 0 ##########")
		print_tensors(sess, [reach_probabilities_lvl0])
		print_tensors(sess, [node_to_IS_lvl0, node_types_lvl0, utilities_lvl0, IS_acting_players_lvl0, IS_strategies_lvl0])
		print("########## Level 1 ##########")
		print_tensors(sess, [node_to_IS_lvl1, node_types_lvl1, utilities_lvl1, IS_acting_players_lvl1, IS_strategies_lvl1])
		print("########## Level 2 ##########")
		print_tensors(sess, [node_to_IS_lvl2, node_types_lvl2, utilities_lvl2, IS_acting_players_lvl2, IS_strategies_lvl2])
		print("########## Level 3 ##########")
		print_tensors(sess, [node_types_lvl3, utilities_lvl3])