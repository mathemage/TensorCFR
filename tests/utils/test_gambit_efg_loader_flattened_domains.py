import os
import unittest

import numpy as np

from src.commons import constants as common_constants
from src.utils import gambit_flattened_domains


class TestGambitLoaderDomain01(unittest.TestCase):
	def setUp(self):
		domain01_efg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc','domain01_via_gambit.efg')

		self.domain = gambit_flattened_domains.Loader(domain01_efg)

	def test_utilities_level_0(self):
		expected_output = [common_constants.NON_TERMINAL_UTILITY]
		np.testing.assert_array_equal(self.domain.utilities[0], expected_output)

	def test_utilities_level_1(self):
		expected_output = [common_constants.NON_TERMINAL_UTILITY] * 5
		np.testing.assert_array_equal(self.domain.utilities[1], expected_output)

	def test_utilities_level_2(self):
		expected_output = [
			common_constants.NON_TERMINAL_UTILITY,
			common_constants.NON_TERMINAL_UTILITY,
			30,
			common_constants.NON_TERMINAL_UTILITY,
			common_constants.NON_TERMINAL_UTILITY,
			common_constants.NON_TERMINAL_UTILITY,
			common_constants.NON_TERMINAL_UTILITY,
			common_constants.NON_TERMINAL_UTILITY,
			common_constants.NON_TERMINAL_UTILITY,
			130,
			common_constants.NON_TERMINAL_UTILITY,
			common_constants.NON_TERMINAL_UTILITY]
		np.testing.assert_array_equal(self.domain.utilities[2], expected_output)

	def test_utilities_level_3(self):
		expected_output = [10, 20, 30, 40, 70, 80, 90, 100, 130, 140, 150, 160, 190, 200, 210, 220, 270, 280, 290, 300]
		np.testing.assert_array_equal(self.domain.utilities[3], expected_output)

	def test_infoset_acting_players_level_0(self):
		expected_output = [0]
		np.testing.assert_array_equal(self.domain.infoset_acting_players[0], expected_output)

	def test_infoset_acting_players_level_1(self):
		expected_output = [1, 2, 2, 0]
		np.testing.assert_array_equal(self.domain.infoset_acting_players[1], expected_output)

	def test_infoset_acting_players_level_2(self):
		expected_output = [1, 2, 1, 2, 0, 1, 2]
		np.testing.assert_array_equal(self.domain.infoset_acting_players[2], expected_output)

	def test_initial_infoset_strategies_level_0(self):
		expected_output = [[0.5, 0.25, 0.1, 0.1, 0.05]]
		np.testing.assert_array_equal(self.domain.initial_infoset_strategies[0], expected_output)

	def test_initial_infoset_strategies_level_1(self):
		expected_output = [[0.33333333, 0.33333333, 0.33333333],
						   [0.5, 0.5, common_constants.IMAGINARY_PROBABILITIES],
						   [0.5, 0.5, common_constants.IMAGINARY_PROBABILITIES],
						   [0.33333333, 0.33333333, 0.33333333]]
		np.testing.assert_array_almost_equal(self.domain.initial_infoset_strategies[1], expected_output, 0.005)

	def test_initial_infoset_strategies_level_2(self):
		expected_output = [[0.5, 0.5],
		                   [0.5, 0.5],
		                   [0.5, 0.5],
		                   [0.5, 0.5],
		                   [0.1, 0.9],
		                   [0.5, 0.5],
		                   [0.5, 0.5]]
		np.testing.assert_array_equal(self.domain.initial_infoset_strategies[2], expected_output)

	def test_node_to_infoset_level_0(self):
		expected_output = [0]
		np.testing.assert_array_equal(self.domain.node_to_infoset[0], expected_output)

	def test_node_to_infoset_level_1(self):
		expected_output = [0, 1, 2, 2, 3]
		np.testing.assert_array_equal(self.domain.node_to_infoset[1], expected_output)

	def test_node_to_infoset_level_2(self):
		expected_output = [0, 1, common_constants.INFOSET_FOR_TERMINAL_NODES, 2, 2, 3, 4, 3, 4,
		                   common_constants.INFOSET_FOR_TERMINAL_NODES, 5, 6]
		np.testing.assert_array_equal(self.domain.node_to_infoset[2], expected_output)

	def test_number_of_nodes_actions_level_0(self):
		expected_output = [5]
		np.testing.assert_array_equal(self.domain.number_of_nodes_actions[0], expected_output)

	def test_number_of_nodes_actions_level_1(self):
		expected_output = [3, 2, 2, 2, 3]
		np.testing.assert_array_equal(self.domain.number_of_nodes_actions[1], expected_output)

	def test_number_of_nodes_actions_level_2(self):
		expected_output = [2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2]
		np.testing.assert_array_equal(self.domain.number_of_nodes_actions[2], expected_output)

	def test_number_of_nodes_actions_level_3(self):
		expected_output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		np.testing.assert_array_equal(self.domain.number_of_nodes_actions[3], expected_output)