from src.algorithms.tensorcfr_flattened_domains.TensorCFRFlattenedDomains import TensorCFRFlattenedDomains, run_cfr
from src.domains.available_domains import get_domain_by_name


if __name__ == '__main__':
	domain = get_domain_by_name("GP_cards4x3_222_gambit_flattened")
	tensorcfr = TensorCFRFlattenedDomains(domain)
	run_cfr(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)   # TODO verify the results (final average strategies) via `gtlibrary`