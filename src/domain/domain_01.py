#!/usr/bin/env python3

import tensorflow as tf

from constants import NON_TERMINAL_UTILITY, INNER_NODE, TERMINAL_NODE, IMAGINARY_NODE
from utils.tensor_utils import print_tensors, masked_assign

# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

########## Level 0 ##########
# I0,0 = {} ... special index - all-1's strategy for counterfactual probability
# I0,1 = { s } ... root state, the opponent acts here
# there are 5 actions in state s
reach_probabilities_lvl0 = tf.Variable(1.0, name="reach_probabilities_lvl0")
state2IS_lvl0 = tf.Variable(1, name="state2IS_lvl0")  # NOTE: the value above is not [1] in order to remove 1
																											# redundant '[]' represented by choice of empty sequence {}
node_types_lvl0 = tf.Variable(INNER_NODE, name="node_types_lvl0")
IS_strategies_lvl0 = tf.Variable([[1.0, 1.0, 1.0, 1.0, 1.0],   # of I0,0
                                  [0.5, .25, 0.1, 0.1, .05]],  # of I0,1
                                 name="IS_strategies_lvl0")
utilities_lvl0 = tf.fill(value=NON_TERMINAL_UTILITY, dims=state2IS_lvl0.shape, name="utilities_lvl0")

########## Level 1 ##########
# I1,0 = { s' } ... special index - all-1's strategy for counterfactual probability
# I1,1 = { s1 }
# I1,2 = { s2, s3 }
# I1,3 = Ic = { s4 } ... chance node
# each state 3 actions
shape_lvl1 = [5]
state2IS_lvl1 = tf.Variable([0, 1, 2, 2, 3], name="state2IS_lvl1")
node_types_lvl1 = tf.Variable([INNER_NODE] * 5, name="node_types_lvl1")
IS_strategies_lvl1 = tf.Variable([[1.0, 1.0, 1.0],   # of I1,0
                                  [0.1, 0.9, 0.0],   # of I1,1
                                  [0.2, 0.8, 0.0],   # of I1,2
                                  [0.3, 0.3, 0.3]],  # of I1,c
                                 name="IS_strategies_lvl1")
utilities_lvl1 = tf.fill(value=NON_TERMINAL_UTILITY, dims=shape_lvl1, name="utilities_lvl1")

########## Level 2 ##########
# I2,0 = { s5, s8, s9, s18 } ... special index - all-1's strategy for counterfactual probability
# I2,1 = { s6  }
# I2,2 = { s11, s14 }
# I2,3 = { s12, s15 } ... chance nodes
# I2,4 = { s19 }
# I2,t = { s7, s10, s13, s16, s17 } ... terminal nodes
# each state 2 actions
shape_lvl2 = [5, 3]
state2IS_lvl2 = tf.Variable([[0, 1, 5],   # s5, s6, s7
                             [0, 0, 5],   # s8, s9, s10
                             [2, 3, 5],   # s11, s12, s13
                             [2, 3, 5],   # s14, s15, s16
                             [5, 0, 4]],  # s17, s18, s19
                            name="state2IS_lvl2")
node_types_lvl2 = tf.Variable([[INNER_NODE, INNER_NODE, TERMINAL_NODE],   # s5, s6, s7
                               [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s8, s9, s10
                               [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s11, s12, s13
                               [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s14, s15, s16
                               [TERMINAL_NODE, INNER_NODE, INNER_NODE]],  # s17, s18, s19
                              name="node_types_lvl2")
IS_strategies_lvl2 = tf.Variable([[1.0, 1.0],   # of I2,0
                                  [0.7, 0.3],   # of I2,1
                                  [0.5, 0.5],   # of I2,2
                                  [0.1, 0.9],   # of I2,3 ... chance player
                                  [0.4, 0.6],   # of I2,4
                                  [0.0, 0.0]],  # of I2,t ... no strategies terminal nodes <- mock-up strategy
                                 name="IS_strategies_lvl2")
utilities_lvl2 = tf.Variable(tf.fill(value=NON_TERMINAL_UTILITY, dims=shape_lvl2))
random_values_lvl2 = tf.random_uniform(shape_lvl2)
mask_terminals_lvl2 = tf.equal(node_types_lvl2, TERMINAL_NODE)
utilities_lvl2 = masked_assign(ref=utilities_lvl2, value=random_values_lvl2, mask=mask_terminals_lvl2,
                               name="utilities_lvl2")

########## Level 3 ##########
# TODO keep track of shapes?
shape_lvl3 = [5, 3, 2]
node_types_lvl3 = tf.Variable(tf.fill(value=TERMINAL_NODE, dims=shape_lvl3))
indices_imaginary_nodes_lvl3 = tf.constant([[0, 2],  # children of s7
                                            [1, 2],  # children of s10
                                            [2, 2],  # children of s13
                                            [3, 2],  # children of s16
                                            [4, 0]], # children of s17
                                           name="indices_imaginary_nodes_lvl3")
node_types_lvl3 = tf.scatter_nd_update(ref=node_types_lvl3, indices=indices_imaginary_nodes_lvl3,
                                       updates=tf.fill(value=IMAGINARY_NODE, dims=indices_imaginary_nodes_lvl3.shape),
                                       name="node_types_lvl3")
utilities_lvl3 = tf.Variable(tf.fill(value=NON_TERMINAL_UTILITY, dims=shape_lvl3))
random_values_lvl3 = tf.random_uniform(shape_lvl3)
mask_terminals_lvl3 = tf.equal(node_types_lvl3, TERMINAL_NODE)
utilities_lvl3 = masked_assign(ref=utilities_lvl3, value=random_values_lvl3, mask=mask_terminals_lvl3,
                               name="utilities_lvl3")


if __name__ == '__main__':
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Level 0 ##########")
		print_tensors(sess, [reach_probabilities_lvl0])
		print_tensors(sess, [state2IS_lvl0, node_types_lvl0, utilities_lvl0, IS_strategies_lvl0])
		print("########## Level 1 ##########")
		print_tensors(sess, [state2IS_lvl1, node_types_lvl1, utilities_lvl1, IS_strategies_lvl1])
		print("########## Level 2 ##########")
		print_tensors(sess, [state2IS_lvl2, node_types_lvl2, utilities_lvl2, IS_strategies_lvl2])
		print("########## Level 3 ##########")
		print_tensors(sess, [node_types_lvl3, utilities_lvl3])
