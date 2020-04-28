import os
import sys
import matplotlib.pyplot as plt
import numpy as np
class Gradient_Descent:
    def __init__(self):
        self.m = 0
        self.b = 0

    def compute_error(self,x,y,m,b):
        total_err = 0
        for i in range(len(x)):
            total_err += (y[i]-((m*x[i])+b))**2
        return total_err/len(x)

    def compute_gradient(self,x,y,m,b):
        m_g = 0
        b_g = 0
        l_r = 0.001
        for i in range(len(x)):
            m_g += -(2/len(x))*(x[i]*(y[i]-(m*x[i]+b)))
            b_g += -(2/len(x))*(y[i]-(m*x[i]+b))
        b_c = b - l_r*b_g
        m_c = m - l_r*m_g
        return b_c,m_c
    def process_grad(self,x,y,epochs):
        err = []
        m,b = self.m,self.b
        for i in range(epochs):
            b,m = self.compute_gradient(x,y,m,b)
           # import pdb;pdb.set_trace()
            err.append(self.compute_error(x,y,m,b))
            self.m,self.b = m,b

        return err[-1],self.m,self.b

x = [11,22,33,44,55,66,77]
y = [12,23,34,45,56,67,78]

gd = Gradient_Descent()
err,m,b = gd.process_grad(x,y,100)
print(err,m,b)
y_f = []
plt.scatter(x,y)
for i in x:
    y_f.append(m*np.float(i)+b)
plt.plot(x,y_f)
plt.show()

