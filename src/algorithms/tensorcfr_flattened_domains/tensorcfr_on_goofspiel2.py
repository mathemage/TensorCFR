from src import utils
from src.algorithms.tensorcfr_flattened_domains.TensorCFRFlattenedDomains import TensorCFRFlattenedDomains, get_cfr_strategies
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	domain = get_domain_by_name("II-GS2_gambit_flattened")
	tensorcfr = TensorCFRFlattenedDomains(domain)
	average_strategies = get_cfr_strategies(
		total_steps=1000,
		tensorcfr_instance=tensorcfr,
		quiet=True,
		# profiling=True,
		delay=0,
		register_strategies_on_step=[0, 1, 500, 999]
	)   # TODO verify the results (final average strategies) via `gtlibrary`
	# export average strategies to JSON
	utils.gtlibrary.export_average_strategies_to_json(
		domain,
		average_strategies,
		"GS2_average_strategies")
