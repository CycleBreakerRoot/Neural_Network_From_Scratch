import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(0)




def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y




X , y = spiral_data(100 , 3)

class layer_dense:
    def __init__(self , n_inputs , n_neurans):
        self.weights = 0.1 * np.random.rand(n_inputs , n_neurans)
        self.baises = np.zeros((1,n_neurans))

    def forward(self , input):
        self.output = np.dot(input , self.weights) + self.baises


class Activation_ReLU:
    def forward(self , input):
        self.output =  np.maximum(0 , input)
    

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs , axis= 1 , keepdims= True))
        parbabilities = exp_values  / np.sum(exp_values , axis= 1  , keepdims= True)
        self.output = parbabilities







dense1 = layer_dense(2 , 3)
act1 = Activation_ReLU()

dense2 = layer_dense(3 , 3)
act2  = Activation_Softmax()

dense1.forward(X)
act1.forward(dense1.output)

dense2.forward(act1.output)
act2.forward(dense2.output)

print(act2.output)


