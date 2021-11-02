#!/usr/bin/python3
# -*- coding: utf-8 -*-

try:

    import sys
    import shlex
    import subprocess
    import argparse
    import logging
    import setuptools
    import os

except ImportError as error:

    print(error)
    print()
    print("1. (optional) Setup a virtual environment: ")
    print("  python3 - m venv ~/Python3env/mltraces ")
    print("  source ~/Python3env/mltraces/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

PATH_TEMP = "data/02_failed_monitors/temp/"
PATH_SOURCE = "data/02_failed_monitors/source/"
PATH_RANKING = "data/02_failed_monitors/ranking/"
PATHS = [PATH_RANKING, PATH_SOURCE , PATH_TEMP ]
DEFAULT_TOP_MONITORS = 1
DEFAULT_TRACE_SOURCE_FILE = 'data/02_failed_monitors/source/00_Collection_TRACE_RES-100.txt'
DEFAULT_FAILED_TRACE_FILE = "data/02_failed_monitors/S0-default-m{:0>2d}.sort_u_1n_3n".format(DEFAULT_TOP_MONITORS)
DEFAULT_MONITOR_RANKING_FILE = "data/02_failed_monitors/ranking/pageRank_crescente.txt"

DEFAULT_VERBOSITY_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'

def print_config(args):

    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    lengths = [len(x) for x in vars(args).keys()]
    max_lenght = max(lengths)

    for k, v in sorted(vars(args).items()):
        message = "\t"
        message +=  k.ljust(max_lenght, " ")
        message += " : {}".format(v)
        logging.info(message)

    logging.info("")

def run_cmd(cmd):
    logging.info("")
    logging.info("Command line : {}".format(cmd))

    # transforma em array por questões de segurança -> https://docs.python.org/3/library/shlex.html
    cmd_array = shlex.split(cmd)
    logging.debug("Command array: {}".format(cmd_array))
    subprocess.run(cmd_array, check=False)


def read_ranking_monitors(file_name, filter0, filter1):
    monitors0 = []
    monitors1 = []
    f = open(file_name, 'r')
    for l in f:
        for word in l.split("\t"):
            if filter0 in word:
                monitors0.append(word.split('\n')[0])
            elif filter1 in word:
                monitors1.append(word.split('\n')[0])
            else:
                logging.debug("word: {}".format(word))

    return monitors0, monitors1


def main():
    print("Creating the structure of directories...")

    for path in PATHS:
        if not os.path.exists(path):
            cmd = "mkdir -p {}".format(path)
            print("path: {} cmd: {}".format(path, cmd))
            cmd_array = shlex.split(cmd)
            subprocess.run(cmd_array, check=True)

    print("done.")
    print("")

    parser = argparse.ArgumentParser(description='Monitor failure injection (mfi)')

    help_msg = 'input source file in the TRACE format (.zip).'
    parser.add_argument("--source", "-s", type=str, help=help_msg, default=DEFAULT_TRACE_SOURCE_FILE)

    help_msg = 'input ranking file.'
    parser.add_argument("--ranking", "-r", type=str, help=help_msg, default=DEFAULT_MONITOR_RANKING_FILE)

    help_msg = 'output file in the SNAPSHOT format.'
    parser.add_argument("--output", "-o", type=str, help=help_msg, default=DEFAULT_FAILED_TRACE_FILE)

    help_msg = 'number of monitors.'
    parser.add_argument("--number", "-n", type=int, help=help_msg, default=DEFAULT_TOP_MONITORS)

    help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_VERBOSITY_LEVEL, type=int)

    help_msg = 'delete temporary file in the TRACE format.'
    parser.add_argument("--delete", "-d", dest='delete', help=help_msg, action='store_true')

    args = parser.parse_args()

    if args.verbosity == logging.DEBUG:
        logging.basicConfig(format='%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s',
                            datefmt=TIME_FORMAT, level=args.verbosity)

    else:
        logging.basicConfig(format='%(message)s', datefmt=TIME_FORMAT, level=args.verbosity)

    # if args.output == DEFAULT_FAILED_SWARM_FILE :
    #     args.output = "data/02_failed_monitors/S0m{:0>2d}.sort_u_1n_3n".format(args.top)
    #


    # files = ["00_Collection_TRACE_RES-100.zip",
    #          "01_Aerofly_TRACE_RES-100.zip",
    #          "02_Increibles_TRACE_RES-100.zip",
    #          "03_Happytime_TRACE_RES-100.zip",
    #          "04_Star_TRACE_RES-100.zip",
    #          "05_Mission_TRACE_RES-100.zip"]
    # try:
    #     #support for index usage
    #     i = int(args.source)
    #     args.source = "/home/mansilha/Research/acdc/{}".format(files[i])
    #     args.ranking = "data/data/02_failed_monitors/ranking/ranking_count_0{}.txt".format(i)
    # except:
    #     pass

    print_config(args)

    monitors0, monitors1 = read_ranking_monitors(args.ranking, "11-30_17-00-00", "11-30_17-15-00")

    monitors0 = monitors0[:args.number]
    print("MONITORS 0")
    print("\n".join(monitors0))

    monitors1 = monitors1[:args.number]
    print("MONITORS 1")
    print("\n".join(monitors1))

    monitors_all = []
    for m in monitors0: monitors_all.append(m)
    for m in monitors1: monitors_all.append(m)

    ranking_file_name = args.ranking.split("/")[-1].split(".")[0]
    source_file_name = args.source.split("/")[-1]
    temp_file_name = "{}/{}_TRACE".format(PATH_TEMP, source_file_name)
    cmd = "rm -f {}".format(temp_file_name)
    run_cmd(cmd)

    for m in monitors_all:
        cmd = "zcat {} | grep {} >> {}".format(args.source, m, temp_file_name)
        logging.info("cmd: {}".format(cmd))
        os.system(cmd)

    cmd = "./convert_trace_to_snapshot.sh -f {} -o {}".format(temp_file_name, args.output)
    run_cmd(cmd)

    if args.delete:
        cmd = "rm -f {}".format(temp_file_name)
        run_cmd(cmd)


if __name__ == '__main__':
    sys.exit(main())

