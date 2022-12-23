from rnn_init import Gate_input_forward, memory_forward, memory_backward, Gate_output_backward, Tanh_forward, \
    Tanh_backward


class RNNLayer:
    def forward_layers(self, x, prev_s, U, W, V):
        self.mulu = Gate_input_forward(U, x)
        self.mulw = Gate_input_forward(W, prev_s)
        self.add = memory_forward(self.mulw, self.mulu)
        self.s = Tanh_forward(self.add)
        self.mulv = Gate_input_forward(V, self.s)

    def backward_layers(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward_layers(x, prev_s, U, W, V)
        dV, dsv = Gate_output_backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = Tanh_backward(self.add, ds)
        dmulw, dmulu = memory_backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = Gate_output_backward(W, prev_s, dmulw)
        dU, dx = Gate_output_backward(U, x, dmulu)
        return dprev_s, dU, dW, dV
