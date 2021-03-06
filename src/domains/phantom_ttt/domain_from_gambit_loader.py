import os
import tensorflow as tf

from src.domains.Domain import Domain
from src.utils.other_utils import activate_script


def get_domain_phantom_ttt():
	path_to_domain_filename = os.path.join(
			os.path.dirname(
					os.path.abspath(
							__file__)
			),
			'..',
			'..',
			'..',
			'doc',
			'phantom_ttt',
			'phantom_ttt.efg'
	)
	return Domain.init_from_gambit_file(path_to_domain_filename, domain_name="phantom_ttt_via_gambit")


if __name__ == '__main__' and activate_script():
	phantom_ttt = get_domain_phantom_ttt()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		phantom_ttt.print_domain(sess)
