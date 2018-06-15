#!/usr/bin/env python3
import copy
import numpy as np

from src.commons import constants

from src.utils.gambit import Parser



class NotAcceptableFormatException(Exception):
	pass


class NotRecognizedPlayersException(Exception):
	pass


class NotRecognizedTreeNodeException(Exception):
	pass


class NotImplementedFormatException(Exception):
	pass


class TreeNode:
	def __init__(self, level=None, action_index=None):
		self.level = level
		self.action_index = action_index


class InformationSetManager:
	def __init__(self, level, number_of_information_sets, is_terminal_node_present=False):
		self.level = level
		self.number_of_information_sets = number_of_information_sets
		self.is_terminal_node_present = is_terminal_node_present

		if self.is_terminal_node_present:
			self.__terminal_node_information_set_index = self.number_of_information_sets
		else:
			self.__terminal_node_information_set_index = None

		self.infoset_cnt = 0
		self.infoset_dict = {}
		self.infoset_node_to_infoset = []
		self.infoset_acting_players_list = []

		self.cnt_player_nodes = 0

		self.flag_set = False
		self.flag_imaginary_node_present = False
		self.flag_terminal_node_present = False

	def add(self, node):
		if node['type'] == constants.GAMBIT_NODE_TYPE_TERMINAL:
			return self.__terminal_node_information_set_index

		if node['infoset_id'] not in self.infoset_dict:
			infoset_index = len(self.infoset_dict)
			self.infoset_dict[node['infoset_id']] = [infoset_index, node['type'], node['tensorcfr_id'],
													 node, len(node['actions'])] # TODO upravit
			self.infoset_acting_players_list.insert(0, node['infoset_id'])
			return infoset_index
		else:
			# self.infoset_dict[node['infoset_id']][4] += len(node['actions'])
			return self.infoset_dict[node['infoset_id']][0]
		# # node type due to _is_imaginary_node_present
		# self.infoset_node_to_infoset.append(node['infoset_id'])
		#
		# if node['infoset_id'] not in self.infoset_dict:
		# 	return_node_to_infoset_value = self.infoset_cnt
		# 	self.infoset_dict[node['infoset_id']] = [return_node_to_infoset_value, node['type'], node['tensorcfr_id'], node]
		# 	self.infoset_acting_players_list.insert(0, node['infoset_id'])
		# 	self.infoset_cnt += 1
		# 	return return_node_to_infoset_value
		# else:
		# 	return_node_to_infoset_value = self.infoset_dict[node['infoset_id']][0]
		# 	return return_node_to_infoset_value

	def make_infoset_acting_players(self, next_level_max_no_actions):
		# if self.flag_set is False and self.level > 0:
		# 	self.flag_imaginary_node_present = self._is_imaginary_node_present(self.level, node_types_)
		# 	self.flag_set = True

		infoset_acting_players_ = []
		current_infoset_strategies_ = []

		# if self.flag_imaginary_node_present:
		# 	self.infoset_dict['imaginary-node'] = [self.infoset_cnt, 'tnode', -1]  # last element - imaginary
		# 	self.infoset_acting_players_list.append('imaginary-node')

		for infoset_id in reversed(self.infoset_acting_players_list):
			infoset_acting_players_.insert(0, self.infoset_dict[infoset_id][2])

			if self.infoset_dict[infoset_id][1] == constants.GAMBIT_NODE_TYPE_PLAYER:
				# TODO: @janrudolf Fix here
				#  This is not the correct way to compute uniform strategies. Normalization is not over all nodes in the next
				#  level. It is over number of children from the information set (alternatively number of children at each node
				#  of the information set). But one also needs to keep in mind imaginary node which should not be counted
				#  towards the normalization sum.
				#
				#  Check out the method `get_infoset_uniform_strategies()` located at line 266 of
				#  `algorithms.tensorcfr.TensorCFR.py`.
				current_infoset_strategy = [np.nan] * next_level_max_no_actions

				action = [float(1 / (self.infoset_dict[infoset_id][4]))] * len(self.infoset_dict[infoset_id][3]['actions'])

				for index, action in enumerate(action):
					current_infoset_strategy[index] = action

				current_infoset_strategies_.append(current_infoset_strategy)

			# [np.nan] * next_level_max_no_actions  # TODO This is a hotfix.

			elif self.infoset_dict[infoset_id][1] == constants.GAMBIT_NODE_TYPE_CHANCE:
				current_infoset_strategy = [np.nan] * next_level_max_no_actions

				for index, action in enumerate(reversed(self.infoset_dict[infoset_id][3]['actions'])):
					current_infoset_strategy[index] = action['probability']

				current_infoset_strategies_.append(current_infoset_strategy)

		if self.is_terminal_node_present:
			current_infoset_strategies_.append([0] * next_level_max_no_actions)

		return [np.asarray(infoset_acting_players_, dtype=constants.INT_DTYPE_NUMPY),
		        np.asarray(current_infoset_strategies_)]


class GambitLoader:

	def __init__(self, efg_file):
		# check if there is a terminal node in any level
		self.__is_terminal_per_level = [False]
		# number of information sets per level
		self.__number_of_information_sets_per_level = [dict()]

		self.gambit_filename = efg_file
		self.nodes = list()
		self.number_of_players = 2

		self.domain_name = ""
		self.actions_per_levels = []
		self.max_actions_per_levels = []

		# number of nodes per level starting from level 0
		self.nodes_per_levels = [1]  # level 0 has always one node
		self.number_of_levels = 0

		# self.terminal_nodes_cnt = 0

		with open(efg_file) as self.gambit_file:
			game_header_line = self.gambit_file.readline()
			game_header = Parser.parse_header(game_header_line)

			if game_header['format'] == 'NFG':
				raise NotImplementedFormatException

			self.domain_name = game_header['name']
			print(game_header)

			self.load()

		print(self.actions_per_levels)
		print(self.nodes_per_levels)
		print(self.max_actions_per_levels)

		print("Is terminal per level")
		print(self.__is_terminal_per_level)
		print("Number of information sets per level")
		print(self.__number_of_information_sets_per_level)

		self.number_of_levels = len(self.nodes_per_levels)
		self.nodes_indexes = copy.deepcopy(self.nodes_per_levels)

		# self.nodes_placement = [None] * self.number_of_levels
		self.utilities = [None] * self.number_of_levels
		self.node_to_infoset = [None] * self.number_of_levels

		for level, number_of_nodes in enumerate(self.nodes_per_levels):
			# set initial  utilities to zeros, will  be filled later
			self.utilities[level] = [0] * number_of_nodes
			# set initial values for node_to_infoset
			self.node_to_infoset[level] = [None] * number_of_nodes


		self.infoset_managers = [
			InformationSetManager(
				level=level,
				number_of_information_sets=self.__number_of_information_sets_per_level[level],
				is_terminal_node_present=self.__is_terminal_per_level[level]
			)
			for level in range(len(self.actions_per_levels) + 1)
		]

		#
		# self.node_to_infoset = [None] * self.number_of_levels
		self.current_infoset_strategies = [None] * self.number_of_levels
		# self.initial_infoset_strategies = [None] * self.number_of_levels  # TODO temporary because of TensorCFR.py
		self.infoset_acting_players = [None] * self.number_of_levels


		# self.positive_cumulative_regrets = [None] * self.number_of_levels
		#
		self.node_types = [None] * (self.number_of_levels + 1)
		# self.utilities = [None] * (self.number_of_levels + 1)
		# self.node_to_infoset = [None] * (self.number_of_levels + 1)
		#
		# for idx in range(len(self.actions_per_levels) + 1):
		# 	self.utilities[idx] = np.ones(self.actions_per_levels[:idx]) * constants.NON_TERMINAL_UTILITY
		# 	self.node_types[idx] = np.ones(self.actions_per_levels[:idx], dtype=np.int) * constants.IMAGINARY_NODE
		# 	self.node_to_infoset[idx] = np.ones(self.actions_per_levels[:idx], dtype=np.int) * TMP_NODE_TO_INFOSET_IMAGINARY_NODE

		with open(efg_file) as self.gambit_file:
			self.load_post()

	def load(self):
		lists_of_information_sets_ids_per_level = [dict()]
		# determines the maximum number of actions per level
		stack_nodes_lvl = [TreeNode(level=0)]

		for cnt, line in enumerate(self.gambit_file):
			if Parser.is_gambit_node(line):
				node = Parser.parse_node(line)
				tree_node = stack_nodes_lvl.pop()

				level = tree_node.level

				if node['type'] != constants.GAMBIT_NODE_TYPE_TERMINAL:
					lists_of_information_sets_ids_per_level[level][node['infoset_id']] = True

					if len(self.actions_per_levels) < (level + 1):
						self.actions_per_levels.append(0)
						self.max_actions_per_levels.append(0)
						self.__is_terminal_per_level.append(False)
						lists_of_information_sets_ids_per_level.append(dict())

					for action in node['actions']:
						new_level = level + 1
						stack_nodes_lvl.append(
							TreeNode(
								level=new_level,
							)
						)

					self.actions_per_levels[level] += len(node['actions'])
					self.max_actions_per_levels[level] = max(len(node['actions']), self.max_actions_per_levels[level])
				else:
					self.__is_terminal_per_level[level] = True
			self.number_of_levels = len(self.actions_per_levels)

		self.nodes_per_levels.extend(self.actions_per_levels)
		self.__number_of_information_sets_per_level = [len(information_sets) for information_sets in lists_of_information_sets_ids_per_level]

	def update_utilities(self, level, action_index, value):
		self.utilities[level][self.placement_indices[level] + action_index] = value

	def update_node_types(self, level, coordinates, value):
		if level == 0:
			self.node_types[level] = value
		else:
			self.node_types[level][tuple(coordinates)] = value

	def update_node_to_infoset(self, level, action_index, value):
		self.node_to_infoset[level][self.placement_indices[level] + action_index] = value

	def load_post(self):
		# a vector of indices to filling vectors
		self.placement_indices = copy.deepcopy(self.nodes_per_levels)
		self.placement_indices[0] = 0
		# stack to safe nodes to visit, init with the root node
		nodes_stack = [TreeNode(level=0, action_index=0)]

		for cnt, line in enumerate(self.gambit_file):
			if Parser.is_gambit_node(line):  # TODO try use yield and get rid of this condition
				current_node = Parser.parse_node(line)
				current_tree_node = nodes_stack.pop()

				node_to_infoset_value = self.infoset_managers[current_tree_node.level].add(current_node)
				self.update_node_to_infoset(current_tree_node.level, current_tree_node.action_index, node_to_infoset_value)

				if current_node['type'] != constants.GAMBIT_NODE_TYPE_TERMINAL:
					# count the number of actions of the current node
					actions_count = len(current_node['actions'])
					# update the index of placement for the next level
					self.placement_indices[current_tree_node.level+1] -= actions_count

					for action_index, action in enumerate(reversed(current_node['actions'])):
						nodes_stack.append(TreeNode(level=current_tree_node.level+1, action_index=action_index))
				else:
					# update utilities for a terminal node
					self.update_utilities(current_tree_node.level, current_tree_node.action_index, current_node['payoffs'][0])

		# for level in range(1, self.number_of_levels):
		# 	self.node_to_infoset[level] = self.infoset_managers[level].make_node_to_infoset(self.node_to_infoset[level])
		print("Number of levels", self.number_of_levels)
		for level, max_action in enumerate(self.max_actions_per_levels):
			[infoset_acting_players_, infoset_strategies] = self.infoset_managers[level].make_infoset_acting_players(
				max_action)
			self.infoset_acting_players[level] = infoset_acting_players_
			self.current_infoset_strategies[level] = infoset_strategies
		# 	self.initial_infoset_strategies[level] = np.array(copy.deepcopy(self.current_infoset_strategies[level]))
		# 	self.cumulative_regrets[level] = np.zeros(infoset_strategies.shape)
		# 	self.positive_cumulative_regrets[level] = np.zeros(infoset_strategies.shape)


if __name__ == '__main__':
	import os
	domain01_efg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'domain01_via_gambit.efg')
	mini_goofspiel_gbt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'mini_goofspiel',
	                                  'mini_goofspiel_via_gtlibrary.gbt')
	# noinspection SpellCheckingInspection
	goofspiel_gbt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'goofspiel',
	                             'IIGS5_s1_bf_ft.gbt')
	poker_gbt = os.path.join(
		os.path.dirname(
			os.path.abspath(
				__file__)
		),
		'..',
		'..',
		'doc',
		'poker',
		'GP_cards2x2_122.gbt'
	)
	gbt_files = [
		domain01_efg,
		# mini_goofspiel_gbt,
		# goofspiel_gbt,
		# poker_gbt,
	]

	domain = GambitLoader(domain01_efg)

	for level in [0,1,2,3]:
		print("LEVEL {}".format(level))

		if level == 3:
			print("node_to_infoset")
			print(domain.node_to_infoset[level])
			print("current_infoset_strategies")
			print(domain.current_infoset_strategies[level])
		else:
			print("node_to_infoset")
			print(domain.node_to_infoset[level])
			print("infoset_acting_players")
			print(domain.infoset_acting_players[level])
			# print(domain.initial_infoset_strategies[level])
			print("current_infoset_strategies")
			print(domain.current_infoset_strategies[level])

	# for gbt_file in gbt_files:
	# 	domain = GambitEFGLoader(gbt_file)
	# 	print("\n>>>>>>>>>> {} <<<<<<<<<<".format(gbt_file))
	# 	for level in range(len(domain.actions_per_levels) + 1):
	# 		print("\n########## Level {} ##########".format(level))
	# 		print(domain.node_types[level])
	# 		print(domain.utilities[level])
	# 		if level < len(domain.actions_per_levels):
	# 			print(domain.node_to_infoset[level])
	# 			print(domain.infoset_acting_players[level])
	# 			print(domain.initial_infoset_strategies[level])
	# 			print(domain.current_infoset_strategies[level])