import matplotlib.pyplot
import numpy as numpy

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import model_from_json
import tensorflow as tf

DEFAULT_NUM = 0.98
tf.get_logger().setLevel('INFO')


class Neural:

    def __init__(self, size_window_left=0, size_window_right=0):

        self.neural_network = None
        self.size_window_left = size_window_left
        self.size_window_right = size_window_right

    def create_neural_network(self, optimizer_function, loss_function, num_layers, num_neurons, lstm_mode, num_cells):

        if lstm_mode:
            input_size = Input(shape=(self.size_window_right + self.size_window_left + 1, 1))
            self.neural_network = LSTM(num_cells, return_sequences=False)(input_size)

        else:
            input_size = Input(shape=(self.size_window_right + self.size_window_left + 1))
            self.neural_network = Dense(num_neurons, )(input_size)

        self.neural_network = Dropout(0.2)(self.neural_network)

        for i in range(num_layers-1):
            self.neural_network = Dense(num_neurons)(self.neural_network)
            self.neural_network = Dropout(0.5)(self.neural_network)

        self.neural_network = Dense(1, activation='sigmoid')(self.neural_network)
        self.neural_network = Model(input_size, self.neural_network)
        self.neural_network.summary()
        self.neural_network.compile(optimizer=optimizer_function, loss=loss_function, metrics=[loss_function])

    def fit(self, x, y, x_validation, y_validation, number_epochs):

        history = self.neural_network.fit(x, y, epochs=number_epochs, verbose=2,
                                          validation_data=(x_validation, y_validation))
        return history

    @staticmethod
    def plotter_error_evaluate(history, label_x, label_y, file_output):

        # matplotlib.pyplot.plot(mean_square_error_training, 'b', marker='^', label="Training Set")
        # matplotlib.pyplot.plot(mean_square_error_evaluate, 'g', marker='o', label="Validation Set")
        matplotlib.pyplot.legend(loc="upper right")
        matplotlib.pyplot.xlabel(label_x)
        matplotlib.pyplot.ylabel(label_y)
        matplotlib.pyplot.savefig(file_output)

    def predict_neural_network(self, x):

        return self.neural_network.predict(x)

    def save_models(self, model_architecture_file, model_weights_file):

        model_json = self.neural_network.to_json()

        with open(model_architecture_file, "w") as json_file:
            json_file.write(model_json)

        self.neural_network.save_weights(model_weights_file)
        print("Saved model {} {}".format(model_architecture_file, model_weights_file))

    def load_models(self, model_architecture_file, model_weights_file):

        json_file = open(model_architecture_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.neural_network = model_from_json(loaded_model_json)
        self.neural_network.load_weights(model_weights_file)
        print("Loaded model {} {}".format(model_architecture_file, model_weights_file))

    def input_adapter_neural_network(self, sample):

        sample_vectorized = [[float(i) for i in sample]]
        #print("sample_vectorized : {}".format(sample_vectorized))
        sample_reshape = numpy.reshape(sample_vectorized, self.size_window_left+self.size_window_right+1)
        #print("sample_reshape : {}".format(sample_reshape))
        return sample_reshape

    def predict(self, x_input, y_output, support_list, threshold):

        list_input_neural_network, list_output_neural_network, list_output_predicted = [], [], []

        for i in range(len(x_input)):

            list_input_neural_network.append(self.input_adapter_neural_network(x_input[i]))
            list_output_neural_network.append(y_output[i])

        predicted = self.predict_neural_network(numpy.asarray(list_input_neural_network))

        for i in range(len(predicted)):

            if predicted[i] > threshold or float(list_output_neural_network[i]) > DEFAULT_NUM:

                if predicted[i] > threshold and not float(list_output_neural_network[i]) > DEFAULT_NUM:

                    support_list[i][2] = 0

                else:

                    support_list[i][2] = 1

                list_output_predicted.append(support_list[i])

        return list_output_predicted

    def deterministic_correction(self, x_input, y_output, support_list):

        list_input_neural_network, list_output_neural_network, list_output_predicted = [], [], []

        for i in range(len(x_input)):
            list_input_neural_network.append(x_input[i])
            list_output_neural_network.append(y_output[i])

        # para cada janela
        # [0,   1, 2, 3] onde 2 é o centro, 1 é a borda anterior e 3 é a borda seguinte
        for i in range(len(list_input_neural_network)):

            #se as duas bordas forem positivos (101, ou 111), a posição central for positiva (010, 110, 010, 011)
            #  adicionar na lista de suporte
            if (list_input_neural_network[i][1] > 0.2 and list_input_neural_network[i][3]>0.2) or float(list_output_neural_network[i]) > DEFAULT_NUM:

                # se as bordas forem positivas e a posição central for negativo
                if (list_input_neural_network[i][1] > 0.2 and list_input_neural_network[i][3]>0.2) and not float(list_output_neural_network[i]) > DEFAULT_NUM:
                    support_list[i][2] = 0
                else:
                    support_list[i][2] = 1

                list_output_predicted.append(support_list[i])

        return list_output_predicted
