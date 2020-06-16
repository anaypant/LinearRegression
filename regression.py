import matplotlib.pyplot as plt
import numpy as np
import math

class LinearRegression():
    def __init__(self, x, y, bench):
        
        self.x = x
        self.y = y
        self.bench = bench
        
        self.pred = []
        self.m = 1
        self.b = 0
        
        self.learn = 0.005
        self.final = []
        self.plt = []
        


    def calc_error(self):
        self.totalError = 0
        self.trueError = 0
        
        for i in range(len(self.x)):
            self.totalError += (self.y[i] - self.pred[i]) ** 2
        
        self.error = float(self.totalError / float(len(self.x)))
        
        return self.totalError / float(len(self.x))

        for a in range(len(self.x)):
            self.trueError += (self.y[i] - self.pred[i])
        
        
        
        

    


    def gradient_descent_runner(self):
        
        self.b_grad = 0
        self.m_grad = 0
        
        N = float(len(self.x))
        
        for i in range(len(self.x)):
            
            self.b_grad += -(2/N) * (self.y[i] - ((self.m * self.x[i]) + self.b))
            self.m_grad += -(2/N) * self.x[i] * (self.y[i] - ((self.m * self.x[i]) + self.b))

        self.m -= self.learn * self.m_grad
        
        self.b -= self.learn * self.b_grad


    def feedforward(self):
        for i in range(len(self.x)):
            self.out = self.x[i] * self.m + self.b
            self.pred.append(self.out)
    

    def backprop(self):
        
        self.calc_error()
        
        self.gradient_descent_runner()



    def plot(self):
        
        plt.scatter(self.x, self.y, color='blue')
        
        plt.plot(self.x, self.final, color='red')
        
        plt.show()
    

    def plot_not_final(self):
        
        plt.ion()
        
        plt.scatter(self.x, self.y, color='blue')
        
        plt.plot(self.x, self.plt, color='red')        
        
        plt.show()
        
        plt.pause(0.01)
        
        plt.cla()
        
        plt.gca()
    
    def predict(self):
        while True:
            self.user = input("\nInput an x value. ")
            self.user = float(self.user)
            self.ret = self.user * self.m + self.b
            print("Expected y value is: " + str(self.ret))

    def train(self, epoch=10000):
        
        for i in range(epoch):
        
            self.feedforward()
        
            self.backprop()
            
            
            if i % 100 == 0:
        
                self.plt = []

        
                for j in range(len(self.x)):
        
                    self.p = (self.x[j] * self.m) + self.b
        
                    self.plt.append(self.p)    
                                
        
                self.plot_not_final()
        
                plt.xlabel("Iteration: " + str(i) + "\nError: " + str(math.sqrt(self.error / len(self.x))))
            
            if (math.sqrt(self.error / len(self.x))) <= self.bench:
                print("True Error: " + str(math.sqrt(self.error / len(self.x))))
                break
        
        
        
        for q in range(len(self.x)):
        
            t_final = self.x[q] * self.m + self.b
        
            self.final.append(t_final)

        
        
        print("Training finished at " + str(i) + " iterations.")
        print("M: " + str(self.m) + '\nB: ' + str(self.b))

        self.predict()