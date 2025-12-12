import numpy as np

# Define RNNCell class
class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = None # Weight between input and hidden
        self.W_hh = None # Weight between hidden and hidden
        self.b_ih = None # Bias for input to hidden
        self.b_hh = None # Bias for hidden to hidden

        self.dW_ih = None # Gradient between input and hidden
        self.dW_hh = None # Gradient between hidden and hidden
        self.db_ih = None # Gradient between input and hidden
        self.db_hh = None # Gradient between hidden and hidden

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh
    
    def zero_grad(self):
        self.dW_ih = np.zeros_like(self.W_ih)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_ih = np.zeros_like(self.b_ih)
        self.db_hh = np.zeros_like(self.b_hh)
    
    def forward(self, x, h_prev_t):
        z = x @ self.W_ih.T + h_prev_t @ self.W_hh.T + self.b_ih + self.b_hh
        h_t = np.tanh(z)
        return h_t
        
    def backward(self, delta, h, h_prev_l, h_prev_t):
        dz = delta * (1 - h ** 2)  # Derivative of tanh
 
        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih += dz.T @ h_prev_l # h_prev_l is x in layer 0, or h from layer below in deeper layers
        self.dW_hh += dz.T @ h_prev_t
        self.db_ih += dz.sum(axis=0) # the derivative of z with respect to a bias term b is one per batch
        self.db_hh += dz.sum(axis=0) 
 
        # 2) Compute dx, dh_prev_t
        dx = dz @ self.W_ih
        dh_prev_t = dz @ self.W_hh
        return dx, dh_prev_t