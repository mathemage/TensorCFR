#!/usr/bin/env python3
import tensorflow as tf


# TODO move to `utils`
def get_parents_from_action_counts(action_counts):
	"""
	Compute tensor `parents` containing the indices to each node's parent in the previous level.

  Args:
    :param action_counts: A 2-D numpy array containing number of actions of each node.

  Returns:
    A corresponding TensorFlow operation (from the computation graph) that contain the index to each node's parent
     (in the level above).
  """
	# TODO add final level!
	parents = [
		tf.zeros_like(
				action_counts[level],
				name="parents_lvl{}".format(level)
		)
		for level in range(len(action_counts))
	]
	return parents


if __name__ == '__main__':
	""" Demonstrate on `domains.hunger_games`:
	
	```
	action_counts:
	[[2],
	 [1, 6],
	 [4],
	 [3, 3, 2, 2],
	 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
	```
	 
	expected `parents`:
	```
	"hunger_games/parents_lvl0:0"
	[0]

	"hunger_games/parents_lvl1:0"
	[0 0]

	"hunger_games/parents_lvl2:0"
	[0]

	"hunger_games/parents_lvl3:0"
	[0 0 0 0]

	"hunger_games/parents_lvl4:0"
	[0 0 0 0 0 0 0 0 0 0]
	```
	"""

	action_counts = [
		[2],
		[1, 6],
		[4],
		[3, 3, 2, 2],
		[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
	]
	parents_ = get_parents_from_action_counts(action_counts)

	from pprint import pprint
	from src.utils.tensor_utils import print_tensors
	print("action_counts:")
	pprint(action_counts, indent=1, width=50)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print_tensors(sess, parents_)