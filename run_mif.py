#!/usr/bin/python3
# -*- coding: utf-8 -*-

try:
    import sys
    import os
    from tqdm import tqdm
    import argparse
    import logging
    import subprocess
    import shlex
    import datetime
    from logging.handlers import RotatingFileHandler

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

#https://liyin2015.medium.com/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

DEFAULT_OUTPUT_FILE = "default_mfi_ouput.txt"
DEFAULT_APPEND_OUTPUT_FILE = False
DEFAULT_VERBOSITY_LEVEL = logging.INFO
DEFAULT_TRIALS = 1
DEFAULT_START_TRIALS = 0
DEFAULT_CAMPAIGN = "demo"
DEFAULT_VALIDATION_DATASET = "swarm/validation/S1_25.sort_u_1n_3n"
DEFAULT_TRAINING_DATASET = "S2a"
NUM_EPOCHS = 15
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'

PATH_ORIGINAL = "data/01_original"
PATH_TRAINING = "data/01_original"
PATH_FAILED_MON = "data/02_failed_monitors"
PATH_FAILED_PROB =  "data/02_failed_probability"

PATH_RANKING = "data/02_failed_monitors/ranking"
PATH_CORRECTED = "data/03_corrected_monitors"
PATH_MODEL = "models_saved"
PATH_LOG = 'logs/'
PATHS = [PATH_ORIGINAL, PATH_TRAINING, PATH_FAILED_MON, PATH_FAILED_PROB, PATH_CORRECTED, PATH_MODEL, PATH_LOG]

SOURCE_ZIP_FILES = "/home/mansilha/Research/acdc/"
files = ["00_Collection_TRACE_RES-100_from-w5000-to-w6000.zip",
         "00_Collection_TRACE_RES-100.zip",
         "01_Aerofly_TRACE_RES-100.zip",
         "02_Increibles_TRACE_RES-100.zip",
         "03_Happytime_TRACE_RES-100.zip",
         "04_Star_TRACE_RES-100.zip",
         "05_Mission_TRACE_RES-100.zip",
         "02_Increibles_TRACE_RES-100_tail_99pct.zip"]

training_files = []

def get_output_file_name(campaign=DEFAULT_CAMPAIGN):
    return "results_noms22_mfi_{}.txt".format(campaign)


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


def convert_flot_to_int(value):

    if isinstance(value, float):
        value = int(value * 100)

    return value


def get_ranking_filename(dataset):
    filename = "{}/ranking_count_0{}.txt".format(PATH_RANKING, dataset)
    return filename


def get_original_filename(dataset, full=True):
    file_in = ""
    file_out = ""
    if full:
        file_in = "{}/{}".format(PATH_ORIGINAL, dataset)
        file_out = "{}.sort-k-3-3".format(file_in)
        cmd = "sort -k 3,3 {} > {}".format(file_in, file_out)
        logging.debug("running: {}".format(cmd))
        os.system(cmd)

    else:
        file_in = "{}".format(dataset)
        file_out = "{}.sort-k-3-3".format(file_in)

    return file_out


def get_original_unzip_filename(dataset, full=True):

    #print(files)
    #print(files[dataset-1])
    #sys.exit()

    file_out = ""
    if full:
        file_in = get_original_zip_filename(dataset, True)
        file_temp = "{}/S{}.temp".format(PATH_ORIGINAL, dataset)
        file_out = "{}/S{}.sort_u_1n_4n".format(PATH_ORIGINAL, dataset)
        if not os.path.isfile(file_out):
            cmd = "zcat {} > {}".format(file_in, file_temp)
            logging.info("running: {}".format(cmd))
            os.system(cmd)

            cmd = "./convert_trace_to_snapshot.sh -f {} -o {}".format(file_temp, file_out)
            run_cmd(cmd)
        else:
            logging.info("original_filename exists: {}".format(file_out))

    else:
        file_out = "S{}.sort_u_1n_4n".format(dataset)

    return file_out


def get_original_zip_filename(dataset, full=True):

    file = "{}".format(files[dataset])

    if full:
        file = "{}/{}".format(SOURCE_ZIP_FILES, file)

    return file


def get_training_filename(dataset, full=True):
    return get_original_filename(dataset, full)


def get_validation_swarm_file(file_in=DEFAULT_VALIDATION_DATASET):

    file_out = "{}.sort-k-3-3".format(file_in)
    cmd = "sort -k 3,3 {} > {}".format(file_in, file_out)
    logging.debug("running: {}".format(cmd))
    os.system(cmd)

    return file_out


def get_mon_failed_filename(dataset, mif, full=True):

    if mif is None:
        mif = 100 #all monitors

    filename = "S{}m{:0>2d}.sort_u_1n_4n".format(dataset, mif)
    if full:
        filename = "{}/{}".format(PATH_FAILED_MON, filename)
    return filename


def get_prob_failed_filename(dataset, pif, seed, full=True):

    pif = convert_flot_to_int(pif)
    filename = "{}.failed_pif-{:0>2d}_seed-{:0>3d}".format(get_original_filename(dataset, False), pif, seed)

    if full:
        filename = "{}/{}".format(PATH_FAILED_PROB, filename)

    return filename


def get_corrected_filename(dataset, mif, seed, threshold, window, rna, full=True):
    if mif is None:
        mif = 100
    threshold = convert_flot_to_int(threshold)
    #filename = "{}.corrected_threshold-{:0>2d}_window-{:0>2d}".format(get_failed_filename(dataset, pif, seed, False), threshold, window)
    filename = "{}_mfi-{:0>2d}.corrected_{}_threshold-{:0>2d}_window-{:0>2d}_epochs-{:0>4d}".format(get_original_unzip_filename(dataset, False),
                                                                      mif, rna, threshold, window, NUM_EPOCHS)
    if full:
        filename = "{}/{}".format(PATH_CORRECTED, filename)

    return filename


def get_architecture_filename(training_file, dense_layers, window, trial, rna, full=True):

    filename = "model_arch_{}_{}_denselayers-{:0>2d}_window-{}_trial-{:0>2d}.json".format(
        rna, training_file.split("/")[-1], dense_layers, window, trial)

    if full:
        filename = "{}/{}".format(PATH_MODEL, filename)

    return filename


def get_weights_filename(training_file, dense_layers, window, trial, rna, full=True):

    filename = "model_weight_{}_{}_epochs-{:0>4d}_denselayers-{:0>2d}_window-{:0>2d}_trial-{:0>2d}.h5".format(
        rna, training_file.split("/")[-1], NUM_EPOCHS, dense_layers, window, trial)

    if full:
        filename = "{}/{}".format(PATH_MODEL, filename)

    return filename


# Custom argparse type representing a bounded int
# source: https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers
class IntRange:

    def __init__(self, imin=None, imax=None):

        self.imin = imin
        self.imax = imax

    def __call__(self, arg):

        try:
            value = int(arg)

        except ValueError:
            raise self.exception()

        if (self.imin is not None and value < self.imin) or (self.imax is not None and value > self.imax):
            raise self.exception()

        return value

    def exception(self):

        if self.imin is not None and self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer in the range [{self.imin}, {self.imax}]")

        elif self.imin is not None:
            return argparse.ArgumentTypeError(f"Must be an integer >= {self.imin}")

        elif self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer <= {self.imax}")

        else:
            return argparse.ArgumentTypeError("Must be an integer")


def run_cmd(cmd):
    print(cmd)
    logging.info("")
    logging.info("Command line : {}".format(cmd))
    cmd_array = shlex.split(cmd)
    logging.debug("Command array: {}".format(cmd_array))
    subprocess.run(cmd_array, check=True)


class Campaign():

    def __init__(self, datasets, pifs, dense_layers, thresholds, rnas, windows):
        self.datasets = datasets
        self.dense_layers = dense_layers
        self.thresholds = thresholds
        self.pifs = pifs
        self.rnas = rnas
        self.windows = windows


def create_monitor_injected_fail_file(dataset, mif):
    '''
        -i      input file
        -o      output file
        -r      random number generator seed
        -p      failure probability - must be expressed between [0,100]

    :param dataset:
    :param pif:
    :param trial:
    :return:
    '''
    if mif is None:
        mif = 100 #all monitors

    cmd = "python3 emulate_monitor_failures.py "
    cmd += "--source {} ".format(get_original_zip_filename(dataset))
    cmd += "--output {} ".format(get_mon_failed_filename(dataset, mif))
    cmd += "--ranking {} ".format(get_ranking_filename(dataset))
    cmd += "--number {} ".format(mif)
    run_cmd(cmd)


def create_probability_injected_fail_file(dataset, pif, trial):
    '''
        -i      input file
        -o      output file
        -r      random number generator seed
        -p      failure probability - must be expressed between [0,100]

    :param dataset:
    :param pif:
    :param trial:
    :return:
    '''

    cmd = "./emulate_snapshot_failures.sh "
    cmd += "-i {} ".format(get_original_filename(dataset))
    cmd += "-o {} ".format(get_prob_failed_filename(dataset, pif, trial))
    cmd += "-r {} ".format(trial)
    cmd += "-p {} ".format(convert_flot_to_int(pif))
    run_cmd(cmd)


def check_files(files):
    for f in files:
        if not os.path.isfile(f):
            logging.info("ERROR: file not found! {}".format(f))
            sys.exit(1)

def main():

    print("Creating the structure of directories...")

    for path in PATHS:

        cmd = "mkdir -p {}".format(path)
        print("path: {} cmd: {}".format(path, cmd))
        cmd_array = shlex.split(cmd)
        subprocess.run(cmd_array, check=True)

    print("done.")
    print("")

    parser = argparse.ArgumentParser(description='Torrent Trace Correct - Machine Learning')

    help_msg = 'append output logging file with analysis results (default={})'.format(DEFAULT_APPEND_OUTPUT_FILE)
    parser.add_argument("--append", "-a", default=DEFAULT_APPEND_OUTPUT_FILE, help=help_msg, action='store_true')

    help_msg = "number of trials (default={})".format(DEFAULT_TRIALS)
    parser.add_argument("--trials", "-r", help=help_msg, default=DEFAULT_TRIALS, type=IntRange(1))

    help_msg = "start trials (default={})".format(DEFAULT_START_TRIALS)
    parser.add_argument("--start_trials", "-s", help=help_msg, default=DEFAULT_START_TRIALS, type=IntRange(0))

    help_msg = "Skip training of the machine learning model training?"
    parser.add_argument("--skip_train", "-t", default=False, help=help_msg, action='store_true')

    help_msg = "Campaign [demo, lstm, no-lstm, deterministic] (default={})".format(DEFAULT_CAMPAIGN)
    parser.add_argument("--campaign", "-c", help=help_msg, default=DEFAULT_CAMPAIGN, type=str)

    help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_VERBOSITY_LEVEL, type=int)

    args = parser.parse_args()

    logging_filename = '{}/run_sbrc21_{}.log'.format(PATH_LOG, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    logging_format = '%(asctime)s\t***\t%(message)s'
    # configura o mecanismo de logging
    if args.verbosity == logging.DEBUG:
        # mostra mais detalhes
        logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

    # formatter = logging.Formatter(logging_format, datefmt=TIME_FORMAT, level=args.verbosity)
    logging.basicConfig(format=logging_format, level=args.verbosity)

    # Add file rotating handler, with level DEBUG
    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotatingFileHandler.setLevel(args.verbosity)
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotatingFileHandler)

    # imprime configurações para fins de log
    print_config(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0' #0
    c_demo = Campaign(datasets=[0], dense_layers=[3], thresholds=[.75], pifs=[11], rnas=["lstm_mode"], windows=[11])

    mifs = [20, 17, 16, 12, 11, 10, 9, 8, 7]
    c_comparison = Campaign(datasets=[1], dense_layers=[3], thresholds=[.75], pifs=mifs,  
                            rnas=["lstm_mode","no-lstm_mode"], windows=[11])
   
    campaigns = [c_comparison]

    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" TRAINING ")
    logging.info("##########################################")
    dense_layers_models = {}
    trials = range(args.start_trials, (args.start_trials + args.trials))
    time_start_campaign = datetime.datetime.now()
    training_dataset = "S1a"
    for trial in trials:
        for count_c, c in enumerate(campaigns):
            for dense_layer in c.dense_layers:
                for window in c.windows:
                    for rna in c.rnas:

                        if not (dense_layer, window, trial, rna) in dense_layers_models.keys():
                            logging.info("\tCampaign: {} Layer: {} Window: {} RNA: {}".format(count_c, dense_layer, window, rna))
                            validation_file = "data/01_original/S1a.sort_u_1n_4n" #get_training_filename(training_dataset)
                            #training_file = "data/01_original/02_Increibles_TRACE_RES-100_head_1pct.txt.sort_u_1n_4n"
                            #training_swarm_file = get_training_filename(training_dataset)
                            logging.debug("\tvalidation_file: {}".format(validation_file))

                            # validation_swarm_file = get_validation_swarm_file()
                            training_file = "data/01_original/S2a.sort_u_1n_4n"
                            logging.debug("\ttraining_file: {}".format(training_file))

                            model_architecture_file = get_architecture_filename(training_file, dense_layer, window, trial, rna)
                            logging.debug("\tmodel_architecture_file: {}".format(model_architecture_file))

                            model_weights_file = get_weights_filename(training_file, dense_layer, window, trial, rna)
                            logging.debug("\tmodel_weights_file: {}".format(model_weights_file))

                            dense_layers_models[(dense_layer, window, trial, rna)] = (model_architecture_file, model_weights_file)

                            check_files([training_file])

                            time_start_experiment = datetime.datetime.now()
                            logging.info(
                                "\t\t\t\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))

                            if not args.skip_train and rna != "deterministic":
                                cmd = "python3 main_mif.py "
                                cmd += " --dense_layers {} ".format(dense_layer)
                                cmd += " --training_swarm_file {} ".format(training_file)
                                cmd += " --validation_swarm_file {} ".format(validation_file)
                                cmd += " --seed {} ".format(trial)
                                cmd += " --model_architecture_file {} ".format(model_architecture_file)
                                cmd += " --model_weights_file {} ".format(model_weights_file)
                                cmd += " --size_window_left {} ".format(int(window/2))
                                cmd += " --size_window_right {} ".format(int(window/2))
                                cmd += " --skip_correct "
                                cmd += " --skip_analyse "
                                cmd += " --{} ".format(rna)
                                #cmd += " --mode {} ".format(rna)
                                cmd += " --num_epochs {} ".format(NUM_EPOCHS)
                                run_cmd(cmd)
                                time_end_experiment = datetime.datetime.now()
                                logging.info("\t\t\t\t\t\t\tEnd                : {}".format(
                                    time_end_experiment.strftime(TIME_FORMAT)))
                                logging.info("\t\t\t\t\t\t\tExperiment duration: {}".format(
                                    time_end_experiment - time_start_experiment))

                                check_files([model_architecture_file, model_weights_file])

    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" EVALUATION ")
    logging.info("##########################################")

    count_trial = 1
    for trial in trials:
        logging.info("\tTrial {}/{} ".format(count_trial, len(trials)))
        count_trial += 1
        count_campaign = 1
        for c in campaigns:
            logging.info("\t\tCampaign {}/{} ".format(count_campaign, len(campaigns)))
            count_campaign += 1
            count_dataset = 1
            for dataset in c.datasets:
                logging.info("\t\t\tDatasets {}/{} ".format(count_dataset, len(c.datasets)))
                count_dataset += 1

                original_swarm_file = get_original_unzip_filename(dataset, 30) #30 monitores -> sem falha
                logging.info("\t\t\t#original_swarm_file: {}".format(original_swarm_file))

                count_denselayers = 1
                for dense_layer in c.dense_layers:
                    logging.info("\t\t\t\tDenselayers {}/{} ".format(count_denselayers, len(c.dense_layers)))
                    count_denselayers += 1
                    count_pif = 1
                    for pif in c.pifs:
                        logging.info("\t\t\t\t\tPifs {}/{} ".format(count_pif, len(c.pifs)))
                        count_pif += 1

                        failed_swarm_file = get_mon_failed_filename(dataset, pif)
                        logging.info("\t\t\t#failed_swarm_file: {}".format(failed_swarm_file))
                        if not os.path.isfile(failed_swarm_file):
                            create_monitor_injected_fail_file(dataset, pif)

                        count_rna = 1
                        for rna in c.rnas:
                            logging.info("\t\t\t\t\t\tRNA {}/{} ".format(count_rna, len(c.rnas)))
                            count_rna += 1

                            count_threshold = 1
                            for threshold in c.thresholds:
                                logging.info("\t\t\t\t\t\t\tThreshold {}/{} ".format(count_threshold, len(c.thresholds)))
                                count_threshold += 1

                                count_window = 1
                                for window in c.windows:
                                    logging.info("\t\t\t\t\t\t\t\tWindow {}/{} ".format(count_window, len(c.windows)))
                                    count_window += 1

                                    (model_architecture_file, model_weights_file) = dense_layers_models[ (dense_layer, window, trial, rna)]
                                    corrected_swarm_file = get_corrected_filename(dataset, pif, trial, threshold, window, rna)
                                    time_start_experiment = datetime.datetime.now()
                                    logging.info("\t\t\t\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))

                                    if not args.skip_train:
                                        check_files([model_architecture_file, model_weights_file])

                                    check_files([original_swarm_file, failed_swarm_file])
                                    cmd = "python3 main_mif.py --skip_train "
                                    cmd += " --threshold {} ".format(threshold)
                                    cmd += " --dense_layers {} ".format(dense_layer)

                                    cmd += " --dataset {} ".format(dataset)
                                    cmd += " --seed {} ".format(trial)
                                    cmd += " --size_window_left {} ".format(int(window/2))
                                    cmd += " --size_window_right {} ".format(int(window/2))
                                    cmd += " --model_architecture_file {} ".format(model_architecture_file)
                                    cmd += " --model_weights_file {} ".format(model_weights_file)
                                    cmd += " --analyse_file {} ".format(get_output_file_name(args.campaign))
                                    cmd += " --analyse_file_mode a "
                                    cmd += " --skip_train "
                                    cmd += " --original_swarm_file {} ".format(original_swarm_file)
                                    cmd += " --failed_swarm_file {} ".format(failed_swarm_file)
                                    cmd += " --corrected_swarm_file {} ".format(corrected_swarm_file)
                                    cmd += " --{} ".format(rna)
                                    cmd += " --mode {} ".format(rna)
                                    cmd += " --num_epochs {} ".format(NUM_EPOCHS)
                                    if pif is None:
                                        cmd += " --skip_analyse"
                                        cmd += " --pif 100 "
                                    else:
                                        cmd += " --pif {} ".format(pif)

                                    run_cmd(cmd)
                                    time_end_experiment = datetime.datetime.now()
                                    logging.info("\t\t\t\t\t\t\t\tEnd                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
                                    logging.info("\t\t\t\t\t\t\t\tExperiment duration: {}".format(time_end_experiment - time_start_experiment))

                                    check_files([corrected_swarm_file])

    time_end_campaign = datetime.datetime.now()
    logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))


if __name__ == '__main__':
    sys.exit(main())
