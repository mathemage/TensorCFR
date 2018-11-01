from src import utils
from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.available_domains import get_domain_by_name

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = True

if __name__ == '__main__' and ACTIVATE_FILE:
	domain = get_domain_by_name("IIGS6_gambit_flattened")
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=0
	)
	average_strategies = tensorcfr.cfr_strategies_after_fixed_trunk(
		total_steps=1000,
		delay=0,
		register_strategies_on_step=[0, 250, 500, 750, 999]
	)  # TODO verify the results (final average strategies) via `gtlibrary`

	utils.gtlibrary.export_average_strategies_to_json(
		domain,
		average_strategies,
		"GS6_average_strategies"
	)
