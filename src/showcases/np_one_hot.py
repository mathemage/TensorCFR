#!/usr/bin/env python3

import numpy as np

from src.utils.other_utils import get_one_hot_flattened

N_CLASSES = 3

if __name__ == '__main__':
	features = np.array(
		[
			[0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.21300e-03],
			[0.00000e+00, 0.00000e+00, 0.00000e+00, 2.00000e+00, 3.27100e-03],
			[0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 4.97500e-02],
			[0.00000e+00, 0.00000e+00, 1.00000e+00, 2.00000e+00, 6.64000e-04],
			[0.00000e+00, 0.00000e+00, 2.00000e+00, 0.00000e+00, 1.19360e-02],
			[0.00000e+00, 0.00000e+00, 2.00000e+00, 1.00000e+00, 3.96000e-02],
			[0.00000e+00, 1.00000e+00, 0.00000e+00, 2.00000e+00, 4.25580e-02]
		]
	)
	one_hot_flattened_features = get_one_hot_flattened(features, N_CLASSES)

	print("features:\n{}".format(features))
	print("indices:\n{}".format((features[:, :4]).astype(int)))
	print("one_hot_flattened_features:\n{}".format(one_hot_flattened_features))
