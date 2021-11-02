#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import sys

from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Tahoma']
# rcParams['font.family'] = 'serif'
# rcParams['font.sans-serif'] = ['Times']
rcParams['font.size'] = 12
SHOW = True
def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        #print("height:", height)
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -35),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plota(arquivo, titulo, labels, xlabel, plota_linha, figheight, figwidth,
          accuracy_means, precision_means, recall_means, f1_means,
          accuracy_error, precision_error, recall_error, f1_error,
            accuracy_means_det=None, precision_means_det=None, recall_means_det=None, f1_means_det=None,
            accuracy_error_det=None, precision_error_det=None, recall_error_det=None, f1_error_det=None):



    mapa_cor = plt.get_cmap('tab10')  # carrega tabela de cores conforme dicionário
    mapeamento_normalizado = colors.Normalize(vmin=0, vmax=19)  # mapeamento em 20 cores
    mapa_escalar = cmx.ScalarMappable(norm=mapeamento_normalizado, cmap=mapa_cor)  # lista de cores final
    cores = [mapa_escalar.to_rgba(i*2) for i in range(10)]
    width = 0.20  # the width of the bars

    if accuracy_means_det  is not None:
        mapa_cor = plt.get_cmap('tab20c')  # carrega tabela de cores conforme dicionário
        mapeamento_normalizado = colors.Normalize(vmin=0, vmax=19)  # mapeamento em 20 cores
        mapa_escalar = cmx.ScalarMappable(norm=mapeamento_normalizado, cmap=mapa_cor)  # lista de cores final
        cores = [mapa_escalar.to_rgba(i) for i in range(20)]
        #width = 0.10  # the width of the bars

    x = np.arange(len(labels))  # the label locations
    if accuracy_means_det is not None:
        x = np.arange(len(labels))*2


    fig, ax = plt.subplots()

    if plota_linha is not None:
        print("plota_linha: {}".format(plota_linha))
        acuracia_det = plota_linha['acuracia']  # 0.8611
        plt.axline((0, acuracia_det), (2.25, acuracia_det), color=cores[0])  # acurácia de referência
        precisao_det = plota_linha['precisao']  # 0.9445
        plt.axline((0, precisao_det), (2.25, precisao_det), color=cores[1])  # precisão de referência
        recall_det = plota_linha['recall']  # 0.4899
        plt.axline((0, recall_det), (2.25, recall_det), color=cores[2])  # recall de referência
        f1_det = plota_linha['f1']  # 0.6451
        plt.axline((0, f1_det), (2.25, f1_det), color=cores[3])  # f1 de referência

    if figheight is not None:
        figheight *= 0.393701 #cm to inches
        fig.set_figheight(figheight)
    if figwidth is not None:
        figwidth *= 0.393701 # cm to inches
        fig.set_figwidth(figwidth)

    if accuracy_means_det is None:
        rects1 = ax.bar(x - width*2+width/2, accuracy_means, yerr=accuracy_error, width=width, color=cores[0], label='Accuracy')
        rects2 = ax.bar(x - width*1+width/2, precision_means, yerr=precision_error, width=width, color=cores[1], label='Precision')
        rects3 = ax.bar(x + width*0+width/2, recall_means, yerr=recall_error, width=width, color=cores[2], label='Recall')
        rects4 = ax.bar(x + width*1+width/2, f1_means, yerr=f1_error, width=width, color=cores[3], label='F1')

    else:
        rects1 = ax.bar(x - width * 4 + width / 2, accuracy_means, yerr=accuracy_error, width=width,
                        color=cores[0], label='Accuracy LS')
        rects1d = ax.bar(x - width * 3 + width / 2, accuracy_means_det, yerr=accuracy_error_det, width=width,
                         color=cores[3], label='Accuracy DE')

        rects2 = ax.bar(x - width * 2 + width / 2, precision_means, yerr=precision_error, width=width,
                        color=cores[4], label='Precision LS')
        rects2d = ax.bar(x - width * 1 + width / 2, precision_means_det, yerr=precision_error_det, width=width,
                         color=cores[7], label='Precision DE')

        rects3 = ax.bar(x + width * 0 + width / 2, recall_means, yerr=recall_error, width=width,
                        color=cores[8], label='Recall LS')
        rects3d = ax.bar(x + width * 1 + width / 2, recall_means_det, yerr=recall_error_det, width=width,
                         color=cores[11], label='Recall DE')

        rects4 = ax.bar(x + width * 2 + width / 2, f1_means, yerr=f1_error, width=width,
                        color=cores[12], label='F1 LS')
        rects4d = ax.bar(x + width * 3 + width / 2, f1_means_det, yerr=f1_error_det, width=width,
                         color=cores[15], label='F1 DE')



    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    for i in range(len(x)):
        x[i] = x[i] + 0.0

    ax.set_xticks(x)
    print("x      {}".format(x))
    ax.set_xticklabels(labels)
    print("labels {}".format(labels))
    ax.set_yticks([0.0, 0.5, 1.0])
    #ax.set_ylim(0.5,1)
    ax.legend(loc='lower left', ncol=4, framealpha=1.0)
    #ax.set_xlim(0.5, 9.5)

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    autolabel(ax, rects4)

    if accuracy_means_det is not None:
    #     autolabel(ax, rects1d)
    #     autolabel(ax, rects2d)
    #     autolabel(ax, rects3d)
    #     autolabel(ax, rects4d)
        #plt.axes([0.0, 15.0, 0.0, 1.0])
        pass


    fig.tight_layout()

    #plt.show()

    plt.savefig(arquivo, dpi=300)



def plot2(arquivo, titulo, labels, xlabel, plota_linha, figheight, figwidth,
          accuracy_means, precision_means, recall_means, f1_means,
          accuracy_error, precision_error, recall_error, f1_error,
        accuracy_means_prob=None, precision_means_prob=None, recall_means_prob=None, f1_means_prob=None,
        accuracy_error_prob=None, precision_error_prob=None, recall_error_prob=None, f1_error_prob=None,
            accuracy_means_det=None, precision_means_det=None, recall_means_det=None, f1_means_det=None,
            accuracy_error_det=None, precision_error_det=None, recall_error_det=None, f1_error_det=None):


    mapa_cor = plt.get_cmap('tab20c')  # carrega tabela de cores conforme dicionário
    mapeamento_normalizado = colors.Normalize(vmin=0, vmax=19)  # mapeamento em 20 cores
    mapa_escalar = cmx.ScalarMappable(norm=mapeamento_normalizado, cmap=mapa_cor)  # lista de cores final
    cores = [mapa_escalar.to_rgba(i) for i in range(20)]
    #width = 0.10  # the width of the bars
    width = 0.15  # the width of the bars
    x = np.arange(len(labels))  # the label locations
    if accuracy_means_det is not None:
        x = np.arange(len(labels))*2


    fig, ax = plt.subplots()

    if plota_linha is not None:
        print("plota_linha: {}".format(plota_linha))
        acuracia_det = plota_linha['acuracia']  # 0.8611
        plt.axline((0, acuracia_det), (2.25, acuracia_det), color=cores[0])  # acurácia de referência
        precisao_det = plota_linha['precisao']  # 0.9445
        plt.axline((0, precisao_det), (2.25, precisao_det), color=cores[1])  # precisão de referência
        recall_det = plota_linha['recall']  # 0.4899
        plt.axline((0, recall_det), (2.25, recall_det), color=cores[2])  # recall de referência
        f1_det = plota_linha['f1']  # 0.6451
        plt.axline((0, f1_det), (2.25, f1_det), color=cores[3])  # f1 de referência

    if figheight is not None:
        figheight *= 0.393701 #cm to inches
        fig.set_figheight(figheight)
    if figwidth is not None:
        figwidth *= 0.393701 # cm to inches
        fig.set_figwidth(figwidth)

    bars = 3
    rects1 = ax.bar(x - width * 6 + width / 2, accuracy_means, yerr=accuracy_error, width=width,
                     color=cores[0], label='Accuracy LS')
    rects1p = ax.bar(x - width * 5 + width / 2, accuracy_means_prob, yerr=accuracy_error_prob, width=width,
                     color=cores[1], label='Accuracy PB')
    rects1d = ax.bar(x - width * 4 + width / 2, accuracy_means_det, yerr=accuracy_error_det, width=width,
                     color=cores[2], label='Accuracy ST')

    rects2 = ax.bar(x - width * 3 + width / 2, precision_means, yerr=precision_error, width=width,
                    color=cores[4], label='Precision LS')
    rects2p = ax.bar(x - width * 2 + width / 2, precision_means_prob, yerr=precision_error_prob, width=width,
                     color=cores[5], label='Precision PB')
    rects2d = ax.bar(x - width * 1 + width / 2, precision_means_det, yerr=precision_error_det, width=width,
                     color=cores[6], label='Precision ST')

    rects3 = ax.bar(x + width * 0 + width / 2, recall_means, yerr=recall_error, width=width,
                    color=cores[8], label='Recall LS')
    rects3p = ax.bar(x + width * 1 + width / 2, recall_means_prob, yerr=recall_error_prob, width=width,
                     color=cores[9], label='Recall PB')
    rects3d = ax.bar(x + width * 2 + width / 2, recall_means_det, yerr=recall_error_det, width=width,
                     color=cores[10], label='Recall ST')

    rects4 = ax.bar(x + width * 3 + width / 2, f1_means, yerr=f1_error, width=width,
                    color=cores[12], label='F1 LS')
    rects4p = ax.bar(x + width * 4 + width / 2, f1_means_prob, yerr=f1_error_prob, width=width,
                     color=cores[13], label='F1 PB')
    rects4d = ax.bar(x + width * 5 + width / 2, f1_means_det, yerr=f1_error_det, width=width,
                     color=cores[14], label='F1 ST')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    for i in range(len(x)):
        x[i] = x[i] + 0.0

    ax.set_xticks(x)
    print("x      {}".format(x))
    ax.set_xticklabels(labels)
    print("labels {}".format(labels))
    ax.set_yticks([0.0, 0.5, 1.0])
    #ax.set_ylim(0.5,1)
    ax.legend(loc='upper right', ncol=4, framealpha=1.0, bbox_to_anchor=(.45, 1.27))
    #ax.set_xlim(0.5, 9.5)

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    autolabel(ax, rects4)

    if accuracy_means_det is not None:
    #     autolabel(ax, rects1d)
    #     autolabel(ax, rects2d)
    #     autolabel(ax, rects3d)
    #     autolabel(ax, rects4d)
        #plt.axes([0.0, 15.0, 0.0, 1.0])
        pass

    fig.tight_layout()

    #plt.show()

    plt.savefig(arquivo, dpi=300)

def carrega_arquivo(nome_arquivo, col_media, col_desvio=None):
    '''
    Carrega dados de um arquivo para memória
    :param nome_arquivo: a ser carregado
    :return: n, medias, desvios
    '''
    f = open(nome_arquivo, "r")
    n = []
    medias = []
    desvios = []
    for l in f:
        print("linha: {}".format(l))
        if l[0] != "#":
            label = l.split("\t")[2]
            if "%" in label:
                value = int(label.split("%")[0])
                if value > 100:
                    label = int(value/100)
            n.append(label)

            medias.append(float(l.split("\t")[col_media].replace(",",".")))
            desvios.append(float(l.split("\t")[col_desvio].replace(",",".")))

    f.close()
    return n, medias, desvios


def main():
    # Topologia	        Acurácia	Precisão	Recall	F1
    # [20, 1]	        87%	        95%	        53%	    68%
    # [20, 20, 1]	    88%	        95%	        53%	    72%
    # [20, 20, 20, 1]	88%	        95%	        54%	    69%

    plota_linha = None
    #{}
    # plota_linha['acuracia'] = 0.9572
    # plota_linha['precisao'] = 0.8903
    # plota_linha['recall'] = 0.7011
    # plota_linha['f1'] = 0.7839
    title = None
    plots = []
    plots.append(("sens_threshold.pdf", "data_sens_threshold_lstm.txt",  "Threshold (α)",))
    plots.append(
        ("sens_topology.pdf", "data_sens_topology_lstm.txt", "Number of hidden layers arrangement",))
    plots.append(
        ("sens_window.pdf", "data_sens_window_lstm.txt", "Window size (|X|)",))

    plots.append(
        ("sens_pif.pdf", "data_sens_pif_lstm.txt", "Probabilistic Injected Failure (pif)",))

    for (output, input, xlabel) in plots:
        #LSTM
        coluna = 3
        n_accuracy, accuracy_means, accuracy_error = carrega_arquivo(input, coluna, coluna + 4)

        coluna = 4
        n_precision, precision_means, precision_error = carrega_arquivo(input, coluna, coluna + 4)

        coluna = 5
        n_recall, recall_means, recall_error = carrega_arquivo(input, coluna, coluna + 4)

        coluna = 6
        labels, f1_means, f1_error = carrega_arquivo(input, coluna, coluna + 4)

        #DENSE
        input = input.replace("lstm", "dense")
        coluna = 3
        n_accuracy_det, accuracy_means_det, accuracy_error_det = carrega_arquivo(input, coluna, coluna + 4)

        coluna = 4
        n_precision_det, precision_means_det, precision_error_det = carrega_arquivo(input, coluna, coluna + 4)

        coluna = 5
        n_recall, recall_means_det, recall_error_det = carrega_arquivo(input, coluna, coluna + 4)

        coluna = 6
        labels_det, f1_means_det, f1_error_det = carrega_arquivo(input, coluna, coluna + 4)
        figheight = 13.5
        figwidth = 20

        if "pif" in input:
            figheight = 13.5
            figwidth = 40

        plota(output, title, labels, xlabel, plota_linha, figheight, figwidth,
              accuracy_means, precision_means, recall_means, f1_means,
              accuracy_error, precision_error, recall_error, f1_error,
              accuracy_means_det, precision_means_det, recall_means_det, f1_means_det,
              accuracy_error_det, precision_error_det, recall_error_det, f1_error_det)



    output = "comparison.pdf"
    input = "data_comparison_lstm.txt"
    #xlabel = "Monitor Injected Failure (mif)"
    xlabel = "Probabilistic Injected Failure (pif)"
    # LSTM
    coluna = 3
    n_accuracy, accuracy_means, accuracy_error = carrega_arquivo(input, coluna, coluna + 4)

    coluna = 4
    n_precision, precision_means, precision_error = carrega_arquivo(input, coluna, coluna + 4)

    coluna = 5
    n_recall, recall_means, recall_error = carrega_arquivo(input, coluna, coluna + 4)

    coluna = 6
    labels, f1_means, f1_error = carrega_arquivo(input, coluna, coluna + 4)

    # PROBABILISTIC
    input2 = input.replace("lstm", "probabilistic")
    coluna = 3
    n_accuracy_prob, accuracy_means_prob, accuracy_error_prob = carrega_arquivo(input2, coluna, coluna + 4)

    coluna = 4
    n_precision_prob, precision_means_prob, precision_error_prob = carrega_arquivo(input2, coluna, coluna + 4)

    coluna = 5
    n_recall, recall_means_prob, recall_error_prob = carrega_arquivo(input2, coluna, coluna + 4)

    coluna = 6
    labels_prob, f1_means_prob, f1_error_prob = carrega_arquivo(input2, coluna, coluna + 4)

    # DETERMINISTIC
    input2 = input.replace("lstm", "deterministic")
    coluna = 3
    n_accuracy_det, accuracy_means_det, accuracy_error_det = carrega_arquivo(input2, coluna, coluna + 4)

    coluna = 4
    n_precision_det, precision_means_det, precision_error_det = carrega_arquivo(input2, coluna, coluna + 4)

    coluna = 5
    n_recall, recall_means_det, recall_error_det = carrega_arquivo(input2, coluna, coluna + 4)

    coluna = 6
    labels_det, f1_means_det, f1_error_det = carrega_arquivo(input2, coluna, coluna + 4)

    figheight = 13.5
    figwidth = 40

    plot2(output, title, labels, xlabel, plota_linha, figheight, figwidth,
    accuracy_means, precision_means, recall_means, f1_means,
    accuracy_error, precision_error, recall_error, f1_error,
          accuracy_means_prob, precision_means_prob, recall_means_prob, f1_means_prob,
          accuracy_error_prob, precision_error_prob, recall_error_prob, f1_error_prob,
    accuracy_means_det, precision_means_det, recall_means_det, f1_means_det,
    accuracy_error_det, precision_error_det, recall_error_det, f1_error_det)















    #
    #
    # ####################################
    # # Limiar
    # ####################################
    # arquivo_saida = "limiar.pdf"
    # xlabel = "Threshold (α)"
    # #titulo = "10% de falha, topologia=[20, 20, 20, 1]"
    #
    # arquivo_entrada = "dados_sensi_limiar.txt"
    # coluna = 3
    # n_accuracy, accuracy_means, accuracy_error = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # coluna = 4
    # n_precision, precision_means, precision_error = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # coluna = 5
    # n_recall, recall_means, recall_error = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # coluna = 6
    # labels, f1_means, f1_error = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # plota(arquivo_saida, titulo, labels, xlabel, plota_linha, None, None,
    #       accuracy_means, precision_means, recall_means, f1_means,
    #       accuracy_error, precision_error, recall_error, f1_error)
    #

    # ####################################
    # # PROBABILIDADE DE FALHA
    # ####################################
    # plota_linha = None
    # arquivo_saida = "falha2.pdf"
    # xlabel = "Failure injection probability (pif)"
    # # titulo = "topologia=[20, 20, 20, 1], α=,75"
    #
    # arquivo_entrada = "dados_sensi_pfi.txt"
    # coluna = 3
    # n_accuracy, accuracy_means, accuracy_error = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # coluna = 4
    # n_precision, precision_means, precision_error = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # coluna = 5
    # n_recall, recall_means, recall_error = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # coluna = 6
    # labels, f1_means, f1_error = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    #
    #
    #
    # ##############
    # arquivo_entrada = "dados_sensi_pfi_deterministico.txt"
    # coluna = 3
    # n_accuracy_det, accuracy_means_det, accuracy_error_det = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # coluna = 4
    # n_precision_det, precision_means_det, precision_error_det = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # coluna = 5
    # n_recall, recall_means_det, recall_error_det = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # coluna = 6
    # labels_det, f1_means_det, f1_error_det = carrega_arquivo(arquivo_entrada, coluna, coluna + 4)
    #
    # plota(arquivo_saida, titulo, labels, xlabel, plota_linha, 13.5, 30,
    #       accuracy_means, precision_means, recall_means, f1_means,
    #       accuracy_error, precision_error, recall_error, f1_error,
    #       accuracy_means_det, precision_means_det, recall_means_det, f1_means_det,
    #       accuracy_error_det, precision_error_det, recall_error_det, f1_error_det)




if __name__ == '__main__':
    sys.exit(main())

# Rede Neural	    Limiar	Acurácia	Precisão	Recall	F1
# [20, 20, 20, 1]	0.65	87%	        95%	        54%	    69%
# [20, 20, 20, 1]	0.75	88%	        95%	        54%	    69%
# [20, 20, 20, 1]	0.95	87%	        96%	        50% 	66%