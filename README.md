# Introduction to Deep Learning – Programming Assignments

This repository contains the eight programming exercises I completed independently for the course Introduction to Deep Learning.
All practical work was implemented in Python using PyTorch, focusing on understanding and applying deep learning methods.

The course covers the theoretical foundations of deep learning as well as practical model development across different neural network architectures.

## Exercise 1 – MNIST Classification with PyTorch

Implemented two neural network models using PyTorch:

### Binary classifier – “Is the digit a 2?”

Built a small feed-forward network (784 → hidden → 1 with sigmoid)

Trained using BCELoss for 10 epochs

Evaluated with accuracy, precision, recall, F1-score

Visualized sample predictions and training loss

### Multiclass digit classifier (0–9)

Extended the architecture to two hidden layers with softmax output

Trained using CrossEntropyLoss for 20 epochs

Evaluated via accuracy and confusion matrix

Experimented with learning rates and training duration

## Exercise 7 – “MyTorch”: Implementing an RNN From Scratch

Built core components of a deep learning framework without PyTorch

### Elman RNN Cell

Implemented:

forward pass: affine transforms + tanh activation

backward pass: gradients for weights, biases, input, and previous hidden state

proper gradient accumulation across timesteps

### RNN Phoneme Classifier

Implemented multi-layer RNN forward + backward passes

Used the custom RNN cell for all timesteps

Handled hidden state propagation, sequence processing, and gradient flow

Followed the computation diagrams for both forward and backward passes

This assignment demonstrated understanding of computation graphs, backpropagation through time, and modular neural network design.
