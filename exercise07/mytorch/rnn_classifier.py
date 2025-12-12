import numpy as np
from mytorch.rnn_cell import RNNCell

class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # will store the full hidden state history over all time steps and layers (used for backward)
        self.hiddens = []

        self.rnn = []
        for layer in range(num_layers):
            if layer == 0:
                # first layer takes input_size
                cell = RNNCell(self.input_size, self.hidden_size)
            else:
                # deeper layers take hidden_size as input
                cell = RNNCell(self.hidden_size, self.hidden_size)

            self.rnn.append(cell)
 
    def init_weights(self, rnn_weights, linear_weights):
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)
 
    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0
 
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None # will eventually hold the output from the last time step.
 
        for s in range(seq_len):
            for l in range(self.num_layers):
                if l == 0:
                    inp = x[:, s, :] # layer 0, input is x at time step s
                else:
                    inp = hidden[l - 1, :, :] # layer >0, input is hidden from layer below
                
                hidden[l] = self.rnn[l].forward(inp, hidden[l, :, :])

            self.hiddens.append(hidden.copy())

        logits = self.output_layer.W @ hidden[-1].T + self.output_layer.b
        return logits

    
    def backward(self, delta):
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)
 
        # TODO
        # return dh / batch_size
        #raise NotImplementedError

        for s in reversed(range(seq_len)):
            for l in reversed(range(self.num_layers)):
                if l == 0:
                    h_prev_l = self.x[:, s, :]  # layer 0, input is x at time step s
                else:
                    h_prev_l = self.hiddens[s][l - 1, :, :]  # layer >0, input is hidden from layer below

                dh_t_next = dh[l]  # Save the gradient from next time step (or output layer initially)
                dx, dh_t = self.rnn[l].backward(dh[l], self.hiddens[s + 1][l, :, :], h_prev_l, self.hiddens[s][l, :, :])
                dh[l] = dh_t + dh_t_next # Combine both sources of gradient
                if l > 0:
                    dh[l-1] += dx  # accumulate gradients from upper time step

        return dh / batch_size