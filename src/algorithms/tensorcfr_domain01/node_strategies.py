import tensorflow as tf

from src.domains.domain01.domain_definitions import node_to_infoset, current_infoset_strategies, \
	infoset_acting_players, acting_depth, current_updating_player, shape
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

# noinspection PyShadowingNames
def assign_strategies_to_nodes(infoset_strategies, node_to_infoset, name="assign_strategies_to_nodes",
                               updating_player=None, acting_players=None):
	"""
  Translate 2-D tensor `infoset_strategies` of strategies per information sets to strategies per game states.
  The translation is done based on N-D tensor `states_to_infosets`: each state (indexed by N-D coordinate)
  stores the index of its information set.

  If both `updating_player` and `acting_players` are `None` (default), no masking is used for strategies. Otherwise,
  the `updating_player` acts with probabilities 1 everywhere (for the reach probability in the formula of
  counterfactual values).

  The corresponding TensorFlow operation (in the computation graph) outputs (N+1)-D tensor, which gives
  for every states (indexed by N-D coordinate) the corresponding strategy of its information set. The strategy
  can be read out in the final (N+1)th dimension.

  Args:
    :param infoset_strategies: A 2-D tensor of floats.
    :param node_to_infoset: An N-D tensor of ints.
    :param name: A string to name the resulting tensor operation.
    :param updating_player: The index of the updating player to create for counterfactual probabilities.
    :param acting_players: A tensor of the same shape as `node_to_infoset`, representing acting players per infosets.

  Returns:
    A corresponding TensorFlow operation (from the computation graph).
  """
	if (updating_player is not None) and (acting_players is not None):  # counterfactual reach probabilities
		strategies = tf.where(condition=tf.equal(acting_players, updating_player), x=tf.ones_like(infoset_strategies),
		                      y=infoset_strategies)
	else:
		strategies = infoset_strategies
	return tf.gather(params=strategies, indices=node_to_infoset, name=name)


def get_node_strategies():
	with tf.variable_scope("assign_strategies_to_nodes", reuse=tf.AUTO_REUSE):
		assign_ops = [
			assign_strategies_to_nodes(
					current_infoset_strategies[level],
					node_to_infoset[level],
					name="assign_strategies_to_nodes_lvl{}".format(level),
			) for level in range(acting_depth)
		]
	with tf.variable_scope("node_strategies", reuse=tf.AUTO_REUSE):
		return [
			tf.get_variable(
					name="node_strategies_lvl{}".format(level),
					initializer=assign_ops[level],
			) for level in range(acting_depth)
		]


def read_node_cf_strategies(updating_player=current_updating_player):
	"""
	Here `cf` stands for counterfactual. Simply speaking, on each root->leaf path, the strategies of the updating player
	are set to 1.
	"""
	with tf.variable_scope("node_cf_strategies", reuse=tf.AUTO_REUSE) as node_cf_strategies_scope:
		node_cf_strategies = [
			tf.get_variable(
					name="node_cf_strategies_lvl{}".format(level),
					shape=shape[level + 1],
					dtype=current_infoset_strategies[level].dtype,
			) for level in range(acting_depth)
		]
	# Note: no `reuse=tf.AUTO_REUSE` because these operations exists in 2 variants: 1 per each player
	with tf.variable_scope("set_cf_strategies_to_nodes", reuse=tf.AUTO_REUSE):
		# TODO generate node_cf_strategies_* with tf.where on node_strategies
		assigned_strategies = [
			assign_strategies_to_nodes(
					current_infoset_strategies[level],
					node_to_infoset[level],
					updating_player=updating_player,
					acting_players=infoset_acting_players[level],
					name="assign_cf_strategies_to_nodes_lvl{}".format(level),
			) for level in range(acting_depth)
		]
		set_node_cf_strategies = [
			tf.assign(
					ref=node_cf_strategies[level],
					value=assigned_strategies[level],
			) for level in range(acting_depth)
		]
	with tf.variable_scope(node_cf_strategies_scope, reuse=True):
		with tf.control_dependencies(set_node_cf_strategies):
			return [node_cf_strategies[level].read_value() for level in range(acting_depth)]


def show_strategies(session):
	for level_ in range(acting_depth):
		print("########## Level {} ##########".format(level_))
		print_tensors(session, [
			node_to_infoset[level_],
			current_infoset_strategies[level_],
			node_strategies[level_],
			infoset_acting_players[level_],
			node_cf_strategies_[level_],
		])


if __name__ == '__main__':
	from src.algorithms.tensorcfr_domain01.swap_players import swap_players
	node_strategies = get_node_strategies()
	node_cf_strategies_ = read_node_cf_strategies()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# TODO extract following lines to a UnitTest
		show_strategies(session=sess)
		print("-----------Swap players-----------\n")
		sess.run(swap_players())
		show_strategies(session=sess)
