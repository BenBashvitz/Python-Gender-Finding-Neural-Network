# -*- coding: utf-8 -*-
'''
בן בשביץ
יב 4
'''

import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred.T) ** 2).mean()

class OurNeuralNetwork:
  
  def __init__(self, params):
    # Weights
    self.hidden_w = np.random.random((params, 2))
    self.output_w = np.random.random((2,1))

    # Biases
    self.hidden_b = np.random.random((1,2))
    self.output_b = 1

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h = sigmoid((np.dot(self.hidden_w, x) + self.hidden_b))
    o = sigmoid((np.dot(h, self.output_w) + self.output_b))
    return o

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        sum_h = np.dot(self.hidden_w, x) + self.hidden_b
        h = sigmoid(sum_h)
        y_hidden = h
        
        sum_o = np.dot(h, self.output_w) + self.output_b
        o = sigmoid(sum_o)
        y_pred = o

        #BackPropagation
        error = y_true - y_pred
        d_y_pred = error * deriv_sigmoid(y_pred)
        
        hidden_error = np.dot(d_y_pred, self.output_w.T)
        d_hidden = hidden_error * deriv_sigmoid(y_hidden)
        
        #Updating weights and biases 
        self.hidden_b += np.sum(d_hidden)*learn_rate
        self.output_b += np.sum(d_y_pred)*learn_rate
        self.hidden_w += np.dot(x, d_hidden.T)*learn_rate
        self.output_w += np.dot(y_hidden.T, d_y_pred)*learn_rate
        
        
      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))
    
# Define dataset
data = np.array([
  [25, 6], # Bob
  [-2, -1],  # Alice 
  [-15, -6],# Diana
  [17, 4],# Charlie
])
all_y_trues = np.array([
  0, # Bob
  1, # Alice
  1, # Diana
  0, # Charlie
])


# Train our neural network!
network = OurNeuralNetwork(2)
network.train(data, all_y_trues)
# Test the neural network

female = np.array([-7, -3]) # 128 pounds, 63 inches
male = np.array([20, 2])  # 155 pounds, 68 inches
print(network.feedforward(female)) #needs to output a value close to 1
print(network.feedforward(male)) #needs to output a value close to 0
print(network.feedforward(data[2])) #needs to output a value close to 1

