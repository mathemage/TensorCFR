import numpy as np
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import SMALL_ERROR_TOLERANCE
from src.domains.flattened_domain01_gambit.domain_from_gambit_loader import get_flattened_domain01_from_gambit


class TestNodalExpectedValuesAtTrunkDepth(tf.test.TestCase):
	def setUp(self):
		self.error_tolerance = SMALL_ERROR_TOLERANCE
		self.flattened_domain01 = get_flattened_domain01_from_gambit()

	def test_domain01_lvl2_seed42(self):
		raise NotImplementedError
