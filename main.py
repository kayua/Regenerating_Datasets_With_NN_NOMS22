#!/usr/bin/python3
# -*- coding: utf-8 -*-
import shlex
import subprocess

from analyse import Analyse
from dataset import Dataset
from models.models import Neural

try:

    import sys
    import os
    import datetime
    from tqdm import tqdm
    import argparse
    import logging
    import matplotlib
    import tensorflow
    import numpy
    import keras
    import setuptools


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


DEFAULT_TRAINING_SWARM_FILE = 'swarm/training/S2.sort_u_1n_4n'
DEFAULT_CORRECTED_SWARM_FILE = 'swarm/results/S4.corrected'
DEFAULT_FAILED_SWARM_FILE = "swarm/failed/S4.sorted"
DEFAULT_VALIDATION_SWARM_FILE = "swarm/validation/S1"

DEFAULT_FUNCTION_LOSS = 'mean_squared_error'
DEFAULT_ORIGINAL_SWARM_FILE = 'swarm/original/S4.sorted'
# DEFAULT_TRAINING_SWARM_FILE = 'swarm/training/S2.sort-k-3-3'
# DEFAULT_CORRECTED_SWARM_FILE = 'swarm/results/swarm.corrected'
# DEFAULT_FAILED_SWARM_FILE = "swarm/failed/S1a.sort-k-3-3.failures_10_sort.txt"
# DEFAULT_VALIDATION_SWARM_FILE = "swarm/training/S2.sort-k-3-3"
DEFAULT_OUTPUT_EVOLUTION_ERROR_FIGURES = "evolution/"
DEFAULT_MODEL_ARCHITECTURE_FILE = "models_saved/model_architecture.json"
DEFAULT_MODEL_WEIGHTS_FILE = "models_saved/model_weights.h5"
DEFAULT_PATH_PLOTTERS = "plotter/"

DEFAULT_SIZE_WINDOW_LEFT = 9
DEFAULT_SIZE_WINDOW_RIGHT = 9
DEFAULT_NUMBER_SAMPLES_TRAINING = 200
DEFAULT_NUMBER_EPOCHS = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_DENSE_LAYERS = 4
DEFAULT_NUMBER_NEURONS_PER_LAYER = 20
DEFAULT_LSTM_MODE = False
DEFAULT_NUMBER_CELLS_LSTM = 20
DEFAULT_LEARNING_PATTERNS_PER_ID = False
DEFAULT_OPTIMIZER = 'adam'  # tensorflow.optimizers.Adam(learning_rate=0.001)
DEFAULT_METRIC = "precision"

DEFAULT_VERBOSITY_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_ANALYSE_FILE = "analyse.txt"
DEFAULT_ANALYSE_FILE_MODE = "a+"

def run_cmd(cmd):

    print(cmd)
    logging.info("")
    logging.info("Command line : {}".format(cmd))
    cmd_array = shlex.split(cmd)
    logging.debug("Command array: {}".format(cmd_array))
    subprocess.run(cmd_array, check=True)

# training_swarm_file validation_swarm_file
def training_neural_network(args):
    logging.debug("pass A")
    neural_network = Neural(args.size_window_left, args.size_window_right)

    logging.debug("pass B")
    neural_network.create_neural_network(DEFAULT_OPTIMIZER, DEFAULT_FUNCTION_LOSS, args.dense_layers, args.neurons,
                                         args.lstm_mode, args.cells)
    logging.debug("pass C")
    x_training, y_training, x_validation, y_validation = [], [], [], []
    dataset_training = Dataset(args.size_window_left, args.size_window_right)
    logging.debug("pass D")
    dataset_validation = Dataset(args.size_window_left, args.size_window_right)
    logging.debug("pass E")
    dataset_training.load_samples(args.training_swarm_file)
    logging.debug("pass F: {}".format(args.validation_swarm_file))
    dataset_validation.load_samples(args.validation_swarm_file)
    logging.debug("pass G")

    for _ in tqdm(range(0, args.num_sample_training), desc="Loading training samples"):

        list_snapshots, loop = dataset_training.load_next_peer()
        list_snapshots_fill_gaps = dataset_training.fill_gaps_per_peer(list_snapshots)
        list_snapshots_fill_gaps_border = dataset_training.filling_borders(list_snapshots_fill_gaps)
        x, y, n = dataset_training.get_training_samples(list_snapshots_fill_gaps_border)
        x_training += x
        y_training += y

    for _ in tqdm(range(0, args.num_sample_training), desc="Loading evaluation samples"):

        list_snapshots, loop = dataset_validation.load_next_peer()
        list_snapshots_fill_gaps = dataset_validation.fill_gaps_per_peer(list_snapshots)
        list_snapshots_fill_gaps_border = dataset_validation.filling_borders(list_snapshots_fill_gaps)
        x, y, n = dataset_validation.get_training_samples(list_snapshots_fill_gaps_border)
        x_validation += x
        y_validation += y

    history = neural_network.fit(x_training, y_training, x_validation, y_validation, args.num_epochs)

    if args.lstm_mode != 'lstm-mode':

        list_mean_square_error_fitting = history.history[DEFAULT_FUNCTION_LOSS]
        list_mean_square_error_evaluation = history.history['val_'+DEFAULT_FUNCTION_LOSS]
        write_file_evaluation_fitting(list_mean_square_error_fitting, list_mean_square_error_evaluation, 'average_error_dense.txt')  # Nome do arquivo
        list_precision_fitting = history.history['loss']
        list_precision_evaluation = history.history['val_'+'loss']
        write_file_evaluation_fitting(list_precision_fitting, list_precision_evaluation, 'precision_dense.txt')# Nome do arquivo

    else:

        list_mean_square_error_fitting = history.history[DEFAULT_FUNCTION_LOSS]
        list_mean_square_error_evaluation = history.history['val_' + DEFAULT_FUNCTION_LOSS]

        write_file_evaluation_fitting(list_mean_square_error_fitting, list_mean_square_error_evaluation,
                                      'average_error_lstm.txt')  # Nome do arquivo
        list_precision_fitting = history.history['loss']
        list_precision_evaluation = history.history['val_'+'loss']
        write_file_evaluation_fitting(list_precision_fitting, list_precision_evaluation,
                                      'precision_lstm.txt')  # Nome do arquivo





    #cmd = "python3 plotting_evaluation_fitting.py  "
    #cmd += "--input_file_mse_dense {} ".format('average_error_dense.txt')
    #cmd += "--input_file_precision_dense {}".format('precision_dense.txt')
   #
   # cmd += "--input_file_mse_lstm {} ".format(get_ranking_filename(dataset))
   # cmd += "--input_file_precision_lstm {} ".format(mif)
   #run_cmd(cmd)

    neural_network.save_models(args.model_architecture_file, args.model_weights_file)


# failed_swarm_file corrected_swarm_file


def write_file_evaluation_fitting(fitting, evaluation, file_name):

    output_file = open(DEFAULT_PATH_PLOTTERS+file_name, 'w')

    for i in range(len(fitting)):

        output_file.write(str(fitting[i])+','+str(evaluation[i])+'\n')




def predict_neural_network(args):

    neural_network = Neural(args.size_window_left, args.size_window_right)
    neural_network.load_models(args.model_architecture_file, args.model_weights_file)
    dataset_predict = Dataset(args.size_window_left, args.size_window_right)
    dataset_predict.load_samples(args.failed_swarm_file)
    dataset_predict.create_file_results(args.corrected_swarm_file)

    loop = 1
    print("Predicting the results....")

    while loop:

        list_snapshots, loop = dataset_predict.load_next_peer()
        list_snapshots_fill_gaps = dataset_predict.fill_gaps_per_peer(list_snapshots)
        list_snapshots_fill_gaps_border = dataset_predict.filling_borders(list_snapshots_fill_gaps)
        x, y, support = dataset_predict.get_predict_samples(list_snapshots_fill_gaps_border)

        if len(x) != 0:
            saida_x = neural_network.predict(x, y, support, args.threshold)
            dataset_predict.write_swarm(saida_x)

    dataset_predict.output_file.close()


def predict_deterministic(args):
    neural_network = Neural()
    dataset_predict = Dataset(args.size_window_left, args.size_window_right)
    dataset_predict.load_samples(args.failed_swarm_file)
    dataset_predict.create_file_results(args.corrected_swarm_file)

    loop = 1
    print("Predicting the results....")

    while loop:

        list_snapshots, loop = dataset_predict.load_next_peer()
        list_snapshots_fill_gaps = dataset_predict.fill_gaps_per_peer(list_snapshots)
        list_snapshots_fill_gaps_border = dataset_predict.filling_borders(list_snapshots_fill_gaps)
        x, y, support = dataset_predict.get_predict_samples(list_snapshots_fill_gaps_border)

        if len(x) != 0:
            saida_x = neural_network.deterministic_correction(x, y, support)
            dataset_predict.write_swarm(saida_x)

    dataset_predict.output_file.close()


# original_swarm_file  corrected_swarm_file failed_swarm_file analyse_file
def analyse(args, start_time, size_window_left, size_window_right):
    analise_results = Analyse(args.original_swarm_file, args.corrected_swarm_file, args.failed_swarm_file,
                              args.analyse_file, args.analyse_file_mode, args.dense_layers,
                              args.threshold, args.pif, args.dataset, args.seed, args.lstm_mode, args.mode)
    analise_results.run_analise()
    analise_results.write_results_analyse(start_time, size_window_left, size_window_right)


def print_config(args):
    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    lengths = [len(x) for x in vars(args).keys()]
    max_lenght = max(lengths)

    for k, v in sorted(vars(args).items()):
        if k == "training_swarm_file" and args.skip_train:
            continue
        elif k == "validation_swarm_file" and args.skip_train:
            continue
        elif k == "failed_swarm_file" and args.skip_correct and args.skip_analyse:
            continue
        elif k == "corrected_swarm_file" and args.skip_correct and args.skip_analyse:
            continue
        elif k == "original_swarm_file" and args.skip_analyse:
            continue
        elif k == "analyse_file" and args.skip_analyse:
            continue
        elif k == "analyse_file_mode" and args.skip_analyse:
            continue

        message = "\t"
        message += k.ljust(max_lenght, " ")
        message += " : {}".format(v)
        logging.info(message)

    logging.info("")


def main():
    parser = argparse.ArgumentParser(description='Torrent Trace Correct - Machine Learning')

    help_msg = "File of ground truth."
    parser.add_argument("--original_swarm_file", type=str, help=help_msg, default=DEFAULT_ORIGINAL_SWARM_FILE)

    help_msg = "File of training samples"
    parser.add_argument("--training_swarm_file", type=str, help=help_msg, default=DEFAULT_TRAINING_SWARM_FILE)

    help_msg = "File of correction"
    parser.add_argument("--corrected_swarm_file", type=str, help=help_msg, default=DEFAULT_CORRECTED_SWARM_FILE)

    help_msg = "File of validation"
    parser.add_argument("--validation_swarm_file", type=str, help=help_msg, default=DEFAULT_VALIDATION_SWARM_FILE)

    help_msg = "File of failed swarm"
    parser.add_argument("--failed_swarm_file", type=str, help=help_msg, default=DEFAULT_FAILED_SWARM_FILE)

    help_msg = "Analyse results with statistics"
    parser.add_argument("--analyse_file", type=str, help=help_msg, default=DEFAULT_ANALYSE_FILE)

    help_msg = "Number of dense layers (e.g. 1, 2, 3)"
    parser.add_argument("--dense_layers", type=int, help=help_msg, default=DEFAULT_DENSE_LAYERS)

    help_msg = "Number neurons per layer"
    parser.add_argument("--neurons", type=int, help=help_msg, default=DEFAULT_NUMBER_NEURONS_PER_LAYER)

    help_msg = "Numbers cells(neurons) LSTM"
    parser.add_argument("--cells", type=int, help=help_msg, default=DEFAULT_NUMBER_CELLS_LSTM)

    help_msg = "Number samples for training"
    parser.add_argument("--num_sample_training", type=int, help=help_msg, default=DEFAULT_NUMBER_SAMPLES_TRAINING)

    help_msg = "Number epochs training"
    parser.add_argument("--num_epochs", type=int, help=help_msg, default=DEFAULT_NUMBER_EPOCHS)

    help_msg = "Open mode (e.g. 'w' or 'a')"
    parser.add_argument("--analyse_file_mode", type=str, help=help_msg, default=DEFAULT_ANALYSE_FILE_MODE)

    help_msg = "Full model architecture file"
    parser.add_argument("--model_architecture_file", type=str, help=help_msg, default=DEFAULT_MODEL_ARCHITECTURE_FILE)

    help_msg = "Full model weights file"
    parser.add_argument("--model_weights_file", type=str, help=help_msg, default=DEFAULT_MODEL_WEIGHTS_FILE)

    help_msg = "Left window size"
    parser.add_argument("--size_window_left", type=int, help=help_msg, default=DEFAULT_SIZE_WINDOW_LEFT)

    help_msg = "Right window size"
    parser.add_argument("--size_window_right", type=int, help=help_msg, default=DEFAULT_SIZE_WINDOW_RIGHT)

    help_msg = "i.e. alpha (e.g. 0.5 - 0.95)"
    parser.add_argument("--threshold", type=float, help=help_msg, default=DEFAULT_THRESHOLD)

    help_msg = "Pif (only for statistics)"
    parser.add_argument("--pif", type=float, help=help_msg, default=0.10)

    help_msg = "Dataset (only for statistics)"
    parser.add_argument("--dataset", type=str, help=help_msg, default="Sxxx")

    help_msg = "Seed (only for statistics)"
    parser.add_argument("--seed", type=int, help=help_msg, default=1)

    help_msg = "Activate LSTM mode"
    parser.add_argument("--lstm_mode", dest='lstm_mode', help=help_msg, action='store_true')

    help_msg = "Deactivate LSTM mode"
    parser.add_argument('--no-lstm_mode', dest='lstm_mode', help=help_msg, action='store_false')
    parser.set_defaults(lstm_mode=DEFAULT_LSTM_MODE)

    help_msg = "Skip training of the machine learning model"
    parser.add_argument("--skip_train", "-t", default=False, help=help_msg, action='store_true')

    help_msg = "Set deterministic correction mode"
    parser.add_argument("--deterministic_mode", default=False, help=help_msg, action='store_true')

    help_msg = "Skip correction of the dataset"
    parser.add_argument("--skip_correct", "-c", default=False, help=help_msg, action='store_true')

    help_msg = "Skip analysis of the results"
    parser.add_argument("--skip_analyse", "-a", default=False, help=help_msg, action='store_true')

    help_msg = "Verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_VERBOSITY_LEVEL, type=int)

    help_msg = "Mode"
    parser.add_argument('--mode', type=str, help=help_msg)

    args = parser.parse_args()

    if args.verbosity == logging.DEBUG:
        logging.basicConfig(format="%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
                            datefmt=TIME_FORMAT, level=args.verbosity)

    else:
        logging.basicConfig(format="%(message)s", datefmt=TIME_FORMAT, level=args.verbosity)

    print_config(args)

    start_time = datetime.datetime.now()

    if args.deterministic_mode:
        predict_deterministic(args)
        analyse(args, start_time, args.size_window_left, args.size_window_right)

    else:
        if not args.skip_train:
            training_neural_network(args)

        if not args.skip_correct:
            predict_neural_network(args)

        if not args.skip_analyse:
            analyse(args, start_time, args.size_window_left, args.size_window_right)


if __name__ == '__main__':
    sys.exit(main())
