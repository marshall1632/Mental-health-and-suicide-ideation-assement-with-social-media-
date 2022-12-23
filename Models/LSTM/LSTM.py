import numpy as np

from file_handling import emb_matrix

rng = np.random.default_rng()


class LSTMmodel:
    """
       initializing all the instances of the neural network
    """

    def instance_init(self, d_hidden, d_inputs):
        Weight_forget = rng.standard_normal(size=(d_hidden, d_hidden + d_inputs))
        bias_forget = rng.standard_normal(size=(d_hidden, 1))

        Weight_input = rng.standard_normal(size=(d_hidden, d_hidden + d_inputs))
        bias_input = rng.standard_normal(size=(d_hidden, 1))

        Weight_memory = rng.standard_normal(size=(d_hidden, d_hidden + d_inputs))
        bias_memory = rng.standard_normal(size=(d_hidden, 1))

        Weight_output = rng.standard_normal(size=(d_hidden, d_hidden + d_inputs))
        bias_output = rng.standard_normal(size=(d_hidden, 1))

        Weight_connected_layer = rng.standard_normal(size=(1, d_hidden))
        bias_connected_layer = np.zeros((1, 1))

        parameters = {"Weight_forget": Weight_forget, "bias_forget": bias_forget, "Weight_input": Weight_input,
                      "bias_input": bias_input, "Weight_memory": Weight_memory, "bias_memory": bias_memory,
                      "Weight_output": Weight_output, "bias_output": bias_output,
                      "Weight_connected_layer": Weight_connected_layer,
                      "bias_connected_layer": bias_connected_layer}
        return parameters

    def sigmoid(self, x):
        n = np.exp(np.fmin(x, 0))
        d = (1 + np.exp(-np.abs(x)))
        return n / d

    """
    decides what parts of the old memory cell content need attention and which can be ignored
    """

    def forward_prop_Forget_gate(self, inputs, instance):
        forget_gate = self.sigmoid(np.dot(instance['Weight_forget'], inputs) + instance['bias_forget'])
        return forget_gate

    """
    Input Gate takes the current word embedding and the previous hidden state concatenated together as input
      governs how much of the new data we take into account via the Memory Gate which utilizes the
      to regulate the values flowing through the network.
    """
    def forward_prop_input_gate(self, inputs, instance):
        input_gate = self.sigmoid(np.dot(instance['Weight_input'], inputs) + instance['bias_input'])
        memory_gate = np.tanh(np.dot(instance['Weight_memory'], inputs, out=None) + instance['bias_memory'])
        return input_gate, memory_gate

    """
    Output Gate which takes information from the current word embedding, previous hidden state and the cell state
    which has been updated with information from the forget and input gates to update the value of the hidden state 
    """
    def forward_prop_output_gate(self, inputs, next_state, instance):
        output_gate = self.sigmoid(np.dot(instance['Weight_output'], inputs) + instance['bias_output'])
        next_hidden_state = output_gate * np.tanh(next_state)
        return output_gate, next_hidden_state

    """ 
    activation function such as the sigmoid converts this output to a value between 0 and 1
    this is memory block
    """
    def forwards_prop_fully_connect_layer(self, last_hidden_state, instance):
        connect_layer = (
                np.dot(instance['Weight_connected_layer'], last_hidden_state) + instance['bias_connected_layer'])
        activation = self.sigmoid(connect_layer)
        return activation

    """ 
    summary of forward propagation
    """
    def Forward_proportion(self, vector_X, instance, d_input):
        d_hidden = instance['Weight_forget'].shape[0]
        time_count = len(vector_X)

        previous_hidden_s = np.zeros((d_hidden, 1))
        previous_current_s = np.zeros(previous_hidden_s.shape)

        storage = {'lstm_values': [], 'fully_connected_values': []}

        for t in range(time_count):
            vec = vector_X[t]

            embedding = emb_matrix.get(vec, rng.random(size=(d_input, 1)))
            embedding = embedding.reshape((d_input, 1))

            inputs = np.vstack((previous_hidden_s, embedding))

            forward_gate = self.forward_prop_Forget_gate(inputs, instance)

            input_gate, memory_gate = self.forward_prop_input_gate(inputs, instance)
            in_out = input_gate * memory_gate

            next_cell_state = (forward_gate * previous_current_s) + in_out

            output_gate, next_hidden_s = self.forward_prop_output_gate(inputs, next_cell_state, instance)

            lstm_cache = {"next_hidden_state": next_hidden_s, "next_cell_state": next_cell_state,
                          "previous_hidden_state": previous_hidden_s,
                          "previous_cell_state": previous_current_s, "forget_gate": forward_gate,
                          "input_gate": input_gate,
                          "memory_gate": memory_gate, "output_gate": output_gate, "embed": embedding}
            storage['lstm_values'].append(lstm_cache)

            previous_hidden_s = next_hidden_s
            previous_current_s = next_cell_state

        activation = self.forwards_prop_fully_connect_layer(previous_hidden_s, instance)

        fully_connect_storage = {"activation": activation, "Weight_connected_layer": instance['Weight_connected_layer']}
        storage['fully_connected_values'].append(fully_connect_storage)
        return storage

    """
     initialize gradients of each parameter as arrays made up of zeros with same dimensions as the corresponding
     instances
    """
    def gradients_ini(self, instance):
        grad = {}
        for param in instance.keys():
            grad[f'd{param}'] = np.zeros(instance[param].shape)
        return grad

    """ 
    calculate the gradients in the Forget Gate
    """
    def back_prop_forget_gate(self, d_hidden, inputs, dh_prev, dc_prev, storage, grad, instance):
        d_forget_gate = ((dc_prev * storage["previous_cell_state"] + storage["output_gate"] *
                          (1 - np.square(np.tanh(storage["next_cell_state"])))
                          * storage["previous_cell_state"] * dh_prev) * storage["forget_gate"]
                         * (1 - storage["forget_gate"]))

        grad['dWeight_forget'] += np.dot(d_forget_gate, inputs.T)
        grad['dbias_forget'] += np.sum(d_forget_gate, axis=1, keepdims=True)

        d_hidden_forget_gate = np.dot(instance["Weight_forget"][:, :d_hidden].T, d_forget_gate)
        return d_hidden_forget_gate, grad
    """
     calculate the gradients in the Input Gate and Memory Gate 
     """
    def back_prop_input_gate(self, d_hidden, inputs, hidden_previous, cell_previous, storage, grad, instance):
        d_input = ((cell_previous * storage["memory_gate"] + storage["output_gate"]
                    * (1 - np.square(np.tanh(storage["next_cell_state"])))
                    * storage["memory_gate"] * hidden_previous) * storage["input_gate"] * (1 - storage["input_gate"]))

        d_memory = ((cell_previous * storage["input_gate"] + storage["output_gate"]
                     * (1 - np.square(np.tanh(storage["next_cell_state"])))
                     * storage["input_gate"] * hidden_previous) * (1 - np.square(storage["memory_gate"])))
        grad['dWeight_input'] += np.dot(d_input, inputs.T)
        grad['dWeight_memory'] += np.dot(d_memory, inputs.T)
        grad['dbias_input'] += np.sum(d_input, axis=1, keepdims=True)
        grad['dbias_memory'] += np.sum(d_memory, axis=1, keepdims=True)
        d_hidden_input = np.dot(instance["Weight_input"][:, :d_hidden].T, d_input)
        d_hidden_memory = np.dot(instance["Weight_memory"][:, :d_hidden].T, d_memory)
        return d_hidden_input, d_hidden_memory, grad

    """
     calculate the back prop for output gate gradients
    """
    def back_prop_output_gate(self, d_hidden, inputs, hidden_previous, cell_previous, storage, grad, instance):
        dot = (hidden_previous * np.tanh(storage["next_cell_state"]) * storage["output_gate"] * (
                1 - storage["output_gate"]))
        grad['dWeight_output'] += np.dot(dot, inputs.T)
        grad['dbias_output'] += np.sum(dot, axis=1, keepdims=True)
        d_hidden_output = np.dot(instance["Weight_output"][:, :d_hidden].T, dot)
        return d_hidden_output, grad

    """
     calculate the fully connected layer gradients
    """
    def back_prop_fully_connect_layer(self, point, storage, grad):
        predicted = np.array(storage['fully_connected_values'][0]['activation'])
        point = np.array(point)
        difference = predicted - point
        last_hidden_state = storage['lstm_values'][-1]["next_hidden_state"]
        grad['dWeight_connected_layer'] = np.dot(difference, last_hidden_state.T)
        grad['dbias_connected_layer'] = np.sum(difference)
        weight_activation = storage['fully_connected_values'][0]["Weight_connected_layer"]
        d_last_hidden = np.dot(weight_activation.T, difference)
        return d_last_hidden, grad

    """
     summary of the Backpropagation
    """
    def Back_Propagation(self, point, strorage, d_hidden, d_input, time_count, instance):
        grad = self.gradients_ini(instance)

        d_last_hidden, grad = self.back_prop_fully_connect_layer(point, strorage, grad)

        d_previous_hidden = d_last_hidden
        d_previous_cell = np.zeros(d_previous_hidden.shape)

        for points in reversed(range(time_count)):
            store = strorage['lstm_values'][points]

            inputs = np.concatenate((store["previous_hidden_state"], store["embed"]), axis=0)
            d_hidden_f, grad = self.back_prop_forget_gate(d_hidden, inputs, d_previous_hidden, d_previous_cell, store,
                                                          grad,
                                                          instance)

            d_hidden_input, d_hidden_memory, grad = self.back_prop_input_gate(d_hidden, inputs, d_previous_hidden,
                                                                              d_previous_cell, store, grad, instance)

            d_hidden_output, grad = self.back_prop_output_gate(d_hidden, inputs, d_previous_hidden, d_previous_cell,
                                                               store, grad,
                                                               instance)

            d_previous_hidden = d_hidden_f + d_hidden_input + d_hidden_memory + d_hidden_output
            d_previous_cell = (d_previous_cell * store["forget_gate"] + store["output_gate"]
                               * (1 - np.square(np.tanh(store["next_cell_state"])))
                               * store["forget_gate"] * d_previous_hidden)
        return grad
