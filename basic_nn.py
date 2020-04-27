import tensorflow as tf
import numpy as np

def non_lin(x,derive=False):
    if derive==True:
        return x*(1-x)
    return 1/1+np.exp(-x)

x = np.array([[1,1,0],[1,0,1],[0,0,1]])
y=np.array([[1],[0],[0]])
np.random.seed(1)
syn1 = 2*np.random.random((3,3))-1
syn2 = 2*np.random.random((3,1))-1
for i in range(10000):
    l0 = x
    l1 = non_lin(np.dot(l0,syn1))
    l2 = non_lin(np.dot(l1,syn2))
    l2_err = y - l2
    l2_del = l2_err*non_lin(l2,derive=True)
    l1_err = l2_del.dot(syn2.T)
    l1_del = l1_err*non_lin(l1)
    if i%10 == 0:
        print(str(np.mean(np.abs(l2_err))))

    syn2+=l1.T.dot(l2_del)
    syn1 += l0.T.dot(l1_del)
print(l2)
