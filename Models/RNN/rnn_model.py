import sys
import numpy as np

from layers import RNNLayer
from rnn_init import loss_function, Different_probagation, softmax


class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=64, truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.truncate = truncate

        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        Text = len(x)
        layers = []
        previous_state = np.zeros(self.hidden_dim)
        for t in range(Text):
            layer = RNNLayer()
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            layer.forward_layers(input, previous_state, self.U, self.W, self.V)
            previous_state = layer.s
            layers.append(layer)
        return layers

    def predict(self, x):
        layers = self.forward_propagation(x)
        return [np.argmax(softmax(layer.mulv)) for layer in layers]

    def check_loss(self, x, y):
        assert len(x) == len(y)
        lay = self.forward_propagation(x)
        loss = 0.0
        for i, layer in enumerate(lay):
            loss += loss_function(layer.mulv, y[i])
        return loss / float(len(y))

    def calculate_losss(self, X, Y):
        los = 0.0
        for i in range(len(Y)):
            los += self.check_loss(X[i], Y[i])
        return los / float(len(Y))

    def backward_propagation_cell(self, x, y
                                  ):
        assert len(x) == len(y)
        layer = self.forward_propagation(x)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        T = len(layer)
        previous_state = np.zeros(self.hidden_dim)
        current_state = np.zeros(self.hidden_dim)
        for t in range(0, T):
            diffMulv = Different_probagation(layer[t].mulv, y[t])
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            diffprevious_state, dU_t, dW_t, dV_t = layer[t].backward_layers(input, previous_state,
                                                                            self.U, self.W, self.V,
                                                                            current_state, diffMulv)
            previous_state = layer[t].s
            diffMulv = np.zeros(self.word_dim)
            for i in range(t - 1, max(-1, t - self.truncate - 1), -1):
                input = np.zeros(self.word_dim)
                input[x[i]] = 1
                previous_state_imput = np.zeros(self.hidden_dim) if i == 0 else layer[i - 1].s
                diffprevious_state, dU_i, dW_i, dV_i = layer[i].backward_layers(input, previous_state_imput, self.U,
                                                                                self.W, self.V, diffprevious_state,
                                                                                diffMulv)
                dU_t += dU_i
                dW_t += dW_i
            dV += dV_t
            dU += dU_t
            dW += dW_t
        return dU, dW, dV

    def SGD_gradient(self, x, y, learning_rate):
        dU, dW, dV = self.backward_propagation_cell(x, y)
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW
