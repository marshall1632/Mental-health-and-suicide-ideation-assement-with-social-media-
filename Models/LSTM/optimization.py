import numpy as np


class optimizations:

    def opti(self, d_hidden, d_input, instance):
        vec = {}
        stoch = {}
        for data in instance:
            vec['d' + data] = np.zeros(instance[data].shape)
            stoch['d' + data] = np.zeros(instance[data].shape)
        return vec, stoch
    # updated the weights using the Adam optimization
    def Update_instance(self, instance, grad, vec, stoch, r_learning=0.01, beta=0.9, beta2=0.999):
        for key in instance:
            vec['d' + key] = (beta * vec['d' + key] + (1 - beta) * grad['d' + key])
            stoch['d' + key] = (beta2 * stoch['d' + key] + (1 - beta2) * (grad['d' + key] ** 2))
            instance[key] = (instance[key] - r_learning * vec['d' + key] / np.sqrt(stoch['d' + key] + 1e-8))
        return instance, vec, stoch

    def loss_function(self, A, Y):
        eps = 1e-5
        Loss = (- Y * np.log(A + eps) - (1 - Y) * np.log(1 - A + eps))
        return np.squeeze(Loss)
