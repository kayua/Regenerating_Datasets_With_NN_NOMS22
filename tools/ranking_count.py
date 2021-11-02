import sys
from collections import Counter
from abc import *

import argparse
import logging
import zipfile
from tqdm import tqdm

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_FILE = "test_input.txt"

COLOR_PEER = "green"
COLOR_MONITOR = "red"
COLOR_SNAPSHOT = "cyan"
COLOR_PEER_snapshot = "yellow"


def print_config(args):

    print("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    print("Settings:")
    lengths = [len(x) for x in vars(args).keys()]
    max_lenght = max(lengths)

    for k, v in sorted(vars(args).items()):
        message = "\t"
        message +=  k.ljust(max_lenght, " ")
        message += " : {}".format(v)
        print(message)

    print("")


def print_ranking(monitors, monitors_names):
	sort_monitors = sorted(monitors.items(), key=lambda x: x[1], reverse=False)
	count = 1
	for m in sort_monitors:
		if "_a000000" in monitors_names[m[0]]:
			print('{: 2d}\t{:010d}\t{}\t{}'.format(count, m[1], m[0], monitors_names[m[0]]))
			count += 1


def load_file(zip_file_name, max_lines=0):

	count_lines = 0
	monitors = {}
	monitors_names = {}

	file_name = zip_file_name.split("/")[-1].replace(".zip", ".txt")
	with zipfile.ZipFile(zip_file_name) as thezip:
		with thezip.open(file_name, mode='r') as file:

	#with open(file_name, 'r') as file:
			for line in tqdm(file, desc='Loading lines'):
			#for line in file:
				line = line.decode()
				if line[0] == "#":
					print("skipping line: {}".format(line))
				else:
					snapshot, time, ip_port, peer_id, monitor_id, monitor = line.split(' ')
					monitor_name = monitor[:-1]
					m = "m{}".format(monitor_id)
					p = "p{}".format(peer_id)
					s = "s{}".format(snapshot)
					if m not in monitors.keys():
						monitors[m] = 1
						monitors_names[m] = monitor_name
					else:
						monitors[m] += 1

				count_lines += 1
				if count_lines == max_lines:
					break

	return monitors, monitors_names


def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', default=DEFAULT_FILE, type=str)
	#help_msg = ["{}:{} ".format(i+1,graphs[i].__class__.__name__) for i in range(len(graphs))]
	#help_msg = " ; ".join(help_msg)
	#parser.add_argument('--algorithm', '-a', choices=range(1, len(graphs)+1), help=help_msg, default=1, type=int)
	parser.add_argument('--sizeshow', '-s', help='head e tail monitores no arquivo de saida', default=0, type=int)
	#parser.add_argument('--plot', '-p', help='plot', action='store_true')
	#parser.add_argument('--skip', '-k', help='skip calc', action='store_true')

	# REMOVER DEPOIS
	parser.add_argument('--numberlines', '-n', help='number lines', default=0, type=int) 

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', datefmt=TIME_FORMAT, level=args.log)

	print_config(args)

	monitors, monitors_names = load_file(args.file, args.numberlines)
	print_ranking(monitors, monitors_names)


if __name__ == '__main__':
	main()