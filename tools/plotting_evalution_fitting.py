#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import sys
import csv

from matplotlib import rcParams

DEFAULT_CHARACTER_DELIMITER = ','
DEFAULT_FONT_SIZE_NORMAL = 12
DEFAULT_FONT_SIZE_BIG = 16
DEFAULT_PLOT_FORMATS = ['.', 'v', 'o', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h']
DEFAULT_LINE_STYLES = ['solid', 'dashed']
DEFAULT_MARKERS = ['+', 'x']
DEFAULT_COLOR_TABS = 'tab10'
rcParams['font.size'] = DEFAULT_FONT_SIZE_NORMAL

DEFAULT_INPUT_FILE_AVERAGE_ERROR_DENSE = 'fitting_evaluation/average_error_dense.txt'
DEFAULT_INPUT_FILE_PRECISION_DENSE = 'fitting_evaluation/precision_dense.txt'
DEFAULT_INPUT_FILE_AVERAGE_ERROR_LSTM = 'fitting_evaluation/average_error_lstm.txt'
DEFAULT_INPUT_FILE_PRECISION_LSTM = 'fitting_evaluation/precision_lstm.txt'
DEFAULT_OUTPUT_PLOTTERS = 'plotter/'


def read_data(input_file_name, number_col_x=1):

    data_axis_x = []
    data_axis_y = []

    with open(input_file_name, newline='') as lines:

        line_reader = csv.reader(lines, delimiter=DEFAULT_CHARACTER_DELIMITER)
        counter_epochs = 0

        for line in line_reader:

            if "#" not in line[0]:

                data_axis_y.append(float(line[number_col_x]))
                data_axis_x.append(counter_epochs)
                counter_epochs += 1

    return data_axis_x, data_axis_y


def plotter_data(input_data, title_plotter, output_file, label_axis_y,  label_axis_x):

    color_mapping = plt.get_cmap(DEFAULT_COLOR_TABS)
    normalized_color_map = colors.Normalize(vmin=0, vmax=19)
    mapping_scale_color = cmx.ScalarMappable(norm=normalized_color_map, cmap=color_mapping)

    plotter_color = [mapping_scale_color.to_rgba(i*2) for i in range(10)]

    max_x = 0
    fig, ax = plt.subplots()
    i = 0
    for k in sorted(input_data.keys()):

        data_axis_x, data_axis_y = read_data(input_data[k], i)
        input_data[k] = (data_axis_x, data_axis_y)
        if data_axis_x[-1] > max_x:
            max_x = data_axis_x[-1]
        label = k.split("-")[1]
        ax.plot(data_axis_x, data_axis_y, marker=DEFAULT_MARKERS[i], linestyle=DEFAULT_LINE_STYLES[i], color=plotter_color[i + 3], label=label)
        i += 1

    ax.set_ylabel(label_axis_y, fontsize=DEFAULT_FONT_SIZE_BIG)
    ax.set_xlabel(label_axis_x, fontsize=DEFAULT_FONT_SIZE_BIG)
    ax.set_ylim(0)
    ax.set_xlim(1)
    ax.set_title(title_plotter)

    if "curacy" in label_axis_y:

        legend_localization = 'lower right'

    else:

        legend_localization = 'upper right'

    ax.legend(loc=legend_localization, ncol=1, framealpha=1.0, fontsize=DEFAULT_FONT_SIZE_BIG)
    fig.tight_layout()

    plt.savefig(output_file.replace(".png", ".pdf"), dpi=300)


def second_plotter(files, output):

    color_mapping = plt.get_cmap(DEFAULT_COLOR_TABS)
    normalized_color_mapping = colors.Normalize(vmin=0, vmax=19)
    mapping_scale_color = cmx.ScalarMappable(norm=normalized_color_mapping, cmap=color_mapping)

    plotter_colors = [mapping_scale_color.to_rgba(i*2) for i in range(10)]

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 6))

    for i, file in enumerate(files):

        print("i: {} File: {}".format(i, file))
        data = {}
        data['1-Training'] = file[0]
        data['2-Validation'] = file[0]

        ylabel = file[1]

        for j, k in enumerate(sorted(data.keys())):

            data_axis_x, data_axis_y = read_data(data[k], j)
            label = k.split("-")[1]
            print(data_axis_x)
            print(data_axis_y)
            ax[i].plot(data_axis_x, data_axis_y, marker=DEFAULT_MARKERS[j], linestyle=DEFAULT_LINE_STYLES[j], color=plotter_colors[j + 3], label=label)

        ax[i].set_ylabel(ylabel, fontsize=DEFAULT_FONT_SIZE_BIG)
        ax[i].set_xlabel("Epochs", fontsize=DEFAULT_FONT_SIZE_BIG)
        ax[i].set_ylim(0)
        ax[i].set_xlim(1)

        if i == 1:
            legend_localization = 'lower right'
        else:
            legend_localization = 'upper right'
        ax[i].legend(loc=legend_localization, ncol=1, framealpha=1.0, fontsize=DEFAULT_FONT_SIZE_BIG)

    fig.tight_layout()
    plt.savefig(output, dpi=300)


def main():

    parser = argparse.ArgumentParser(description='Plotter Error')

    help_msg = "Average error dense"
    parser.add_argument("--input_file_mse_dense", type=str, help=help_msg, default=DEFAULT_INPUT_FILE_AVERAGE_ERROR_DENSE)

    help_msg = "Precision dense"
    parser.add_argument("--input_file_precision_dense", type=str, help=help_msg, default=DEFAULT_INPUT_FILE_PRECISION_DENSE)

    help_msg = "Average error LSTM"
    parser.add_argument("--input_file_mse_lstm", type=str, help=help_msg, default='Erro_medio_treino_evolução_lstm.txt')

    help_msg = "Precision LSTM"
    parser.add_argument("--input_file_precision_lstm", type=str, help=help_msg, default='Acurácia_treino_evolução_lstm.txt')

    help_msg = "Output file"
    parser.add_argument("--output_path_plotters", type=str, help=help_msg, default='Acurácia_treino_evolução_lstm.txt')

    args = parser.parse_args()

    files = []

    files.append((args.input_file_mse_dense, "Average error"))
    files.append((args.input_file_precision_dense, "Precision"))
    second_plotter(files, "fitting_dense.pdf")

    files = []
    files.append((DEFAULT_INPUT_FILE_AVERAGE_ERROR_LSTM, "Average error"))
    files.append((DEFAULT_INPUT_FILE_PRECISION_LSTM, "Precision"))
    second_plotter(files, "fitting_lstm.pdf")

    files = []
    files.append((DEFAULT_INPUT_FILE_PRECISION_DENSE, DEFAULT_OUTPUT_PLOTTERS+"fitting_accuracy_dense.pdf", "Precision"))
    files.append((DEFAULT_INPUT_FILE_AVERAGE_ERROR_DENSE, DEFAULT_OUTPUT_PLOTTERS+"fitting_error_dense.pdf", "Average error"))
    files.append((DEFAULT_INPUT_FILE_PRECISION_LSTM, DEFAULT_OUTPUT_PLOTTERS+"fitting_accuracy_lstm.pdf", "Precision"))
    files.append((DEFAULT_INPUT_FILE_AVERAGE_ERROR_LSTM, DEFAULT_OUTPUT_PLOTTERS+"fitting_error_lstm.pdf", "Average error"))

    for file in files:

        data = {}
        data['1-Training'] = file[0]
        data['2-Validation'] = file[0]

        file_output = file[1]
        title = ""
        ylabel = file[2]
        plotter_data(data, title, file_output, ylabel, '')


if __name__ == '__main__':
    sys.exit(main())