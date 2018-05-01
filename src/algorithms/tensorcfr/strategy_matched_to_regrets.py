import tensorflow as tf

from src.domains.domain01.domain_definitions import levels, positive_cumulative_regrets
from src.algorithms.tensorcfr_domain01.regrets import update_positive_cumulative_regrets
from src.algorithms.tensorcfr_domain01.uniform_strategies import get_infoset_uniform_strategies
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_strategy_matched_to_regrets():  # TODO unittest
	update_regrets = update_positive_cumulative_regrets()
	infoset_uniform_strategies = get_infoset_uniform_strategies()
	with tf.control_dependencies(update_regrets):
		with tf.variable_scope("strategies_matched_to_regrets"):
			strategies_matched_to_regrets = [None] * (levels - 1)
			for level in range(levels - 1):
				with tf.variable_scope("level{}".format(level)):
					sums_of_regrets = tf.reduce_sum(
							positive_cumulative_regrets[level].read_value(),
							axis=-1,
							keepdims=True,
							name="sums_of_regrets_lvl{}".format(level)
					)
					normalized_regrets = tf.divide(
							positive_cumulative_regrets[level].read_value(),
							sums_of_regrets,
							name="normalized_regrets_lvl{}".format(level)
					)
					zero_sum_rows = tf.squeeze(
							tf.equal(sums_of_regrets, 0),
							name="zero_sum_rows_lvl{}".format(level)
					)
					# Note: An all-0's row cannot be normalized. Thus, when PCRegrets sum to 0, a uniform strategy is used instead.
					# TODO verify uniform strategy is created (mix of both tf.where branches)
					strategies_matched_to_regrets[level] = tf.where(
							condition=zero_sum_rows,
							x=infoset_uniform_strategies[level],
							y=normalized_regrets,
							name="strategies_matched_to_regrets_lvl{}".format(level)
					)
			return strategies_matched_to_regrets


if __name__ == '__main__':
	strategies_matched_to_regrets_ = get_strategy_matched_to_regrets()
	update_regrets_ = update_positive_cumulative_regrets()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				positive_cumulative_regrets[i],
				strategies_matched_to_regrets_[i],
				strategies_matched_to_regrets_[i],
				update_regrets_[i],
				positive_cumulative_regrets[i],
				strategies_matched_to_regrets_[i],
				strategies_matched_to_regrets_[i],
				update_regrets_[i],
				positive_cumulative_regrets[i],
				strategies_matched_to_regrets_[i],
				strategies_matched_to_regrets_[i],
			])