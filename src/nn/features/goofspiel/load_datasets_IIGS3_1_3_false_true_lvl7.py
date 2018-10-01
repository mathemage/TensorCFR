#!/usr/bin/env python3

# from https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays

import os

import numpy as np
import tensorflow as tf

from src.utils.tf_utils import get_default_config_proto

FEATURES_BASENAME = "IIGS3_1_3_false_true_lvl7"

if __name__ == '__main__':
	script_directory = os.path.dirname(os.path.abspath(__file__))
	npz_filename = "{}/{}_numpy_dataset.npz".format(script_directory, FEATURES_BASENAME)
	with np.load(npz_filename) as data:
		features = data["features"]
		targets = data["targets"]

	# Assume that each row of `features` corresponds to the same row as `labels`.
	assert features.shape[0] == targets.shape[0]

	features_placeholder = tf.placeholder(features.dtype, features.shape)
	targets_placeholder = tf.placeholder(targets.dtype, targets.shape)

	dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, targets_placeholder))
	iterator = dataset.make_initializable_iterator()
	# iterator = dataset.make_one_shot_iterator()   # TODO or this one

	with tf.Session(config=get_default_config_proto()) as sess:
		sess.run(iterator.initializer, feed_dict={features_placeholder: features, targets_placeholder: targets})
