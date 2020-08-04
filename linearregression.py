import matplotlib.pyplot as plt 
import numpy as np
import math


class linearregression():
    def __init__(self, x, y, iteration):
        
        # Initializing some variables
        self.x = x  
        self.y = y 
        self.epoch = iteration
        
        # These are the tuned parameters to make our model better
        self.m = 1 
        self.b = 0

        # This is how fast we want out model to grow
        self.learn = 0.01

    def feedforward(self):

        self.output = []
        
        for i in range(len(self.x)):
            self.output.append(self.x[i] * self.m + self.b)


    def calculate_error(self):
        self.totalError = 0
        self.trueError = 0

        for i in range(len(self.x)):
            self.totalError += (self.y[i] - self.output[i]) ** 2 # The reason we square is for all error values to be positive.

        self.error = float(self.totalError / float(len(self.x)))
        
        return self.totalError / float(len(self.x))

    def gradient_descent(self):
        self.b_grad = 0
        self.m_grad = 0
        N = float(len(self.x))
        
        for i in range(len(self.x)):  
            self.b_grad += -(2/N) * (self.y[i] - ((self.m * self.x[i]) + self.b))
            self.m_grad += -(2/N) * self.x[i] * (self.y[i] - ((self.m * self.x[i]) + self.b))
        
        self.m -= self.learn * self.m_grad
        self.b -= self.learn * self.b_grad

    def backprop(self):

        self.error = self.calculate_error()
        self.gradient_descent()

    def predict(self):
        while True: # This can be taken off if you don't want to predict values forever
            self.user = input("\nInput an x value. ")
            self.user = float(self.user)
            self.ret = self.user * self.m + self.b
            print("Expected y value is: " + str(self.ret))

    def train(self):
        for i in range(self.epoch):
            
            self.feedforward()
            self.backprop()
        self.predict()
