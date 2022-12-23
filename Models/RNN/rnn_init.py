import numpy as np


def Sig_forward(x):
    return 1.0 / (1.0 + np.exp(-x))


def Sig_backward(x, top_diff):
    output = Sig_forward(x)
    return (1.0 - output) * output * top_diff


def Tanh_forward(x):
    return np.tanh(x)


def Tanh_backward(x, top_diff):
    output = Tanh_forward(x)
    return (1.0 - np.square(output)) * top_diff


def Gate_input_forward(W, x):
    return np.dot(W, x)


def Gate_output_backward(W, x, dz):
    dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
    dx = np.dot(np.transpose(W), dz)
    return dW, dx


def memory_forward(x1, x2):
    return x1 + x2


def memory_backward(x1, x2, dz):
    dx1 = dz * np.ones_like(x1)
    dx2 = dz * np.ones_like(x2)
    return dx1, dx2


def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores)


def loss_function(x, y):
    probs = softmax(x)
    return np.log(probs[y])


def Different_probagation(x, y):
    probs = softmax(x)
    probs[y] -= 1.0
    return probs