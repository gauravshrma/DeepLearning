import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np

def sigmoid_func(inputs):
    return (1 / (1 + torch.exp(-inputs)))


def sigmoid_derivative_func(inputs):
    return (torch.exp(-inputs) / (1 + torch.exp(-inputs)) ** 2)


def linear_derivative_func(inputs):
    return (torch.ones(inputs.size[0], inputs.size[1]))


def linear_func(inputs):
    return (inputs)


if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris.data[:, :]  # we only take the first two features.

    for j in range(X.shape[1]):
        X[:, j] = (X[:, j] - np.min(X[:, j])) / (max(X[:, j]) - np.min(X[:, j],0))

    y = iris.target
    y_target = np.zeros((y.shape[0], 1))
    y_target[:, 0] = y

    rand_seq = np.random.permutation(X.shape[0])
    X = X[rand_seq, :]
    y_target = y_target[rand_seq, :]

    y_target_one_hot = np.zeros((150, 3))
    for i in range(X.shape[0]):
        if (y_target[i, 0] == 0):
            y_target_one_hot[i, 0] = 1
        elif (y_target[i, 0] == 1):
            y_target_one_hot[i, 1] = 1
        else:
            y_target_one_hot[i, 2] = 1

    row_X = X.shape[0]
    col_X = X.shape[1]

    # parameters for forward Propogation
    inputs = 4
    hidden_units = 20
    outputs = 3
    y_output = np.zeros(y_target_one_hot.shape)

    w1 = torch.autograd.Variable(torch.randn(inputs, hidden_units,dtype = torch.float64),requires_grad = True)
    w2 = torch.autograd.Variable(torch.randn(hidden_units, outputs,dtype = torch.float64),requires_grad = True)

    #convert numpy arrays to torch
    X = torch.from_numpy(X)
    y_target = torch.from_numpy(y_target)
    y_target_one_hot = torch.from_numpy(y_target_one_hot)
    y_output = torch.from_numpy(y_output)

    # hyperparameters
    learning_rate = 0.001

    i = 0
    while i < 1000:
        for j in range(X.size()[0]):
            hidden_layer_output = sigmoid_func(torch.mm(X[[j], :], w1))
            net_output = torch.mm(hidden_layer_output, w2)
            y_output[j, :] = net_output

            criterion = nn.MSELoss()

            loss = criterion(y_output[j,:], y_target_one_hot[j, :])

            loss.backward(retain_graph=True)
            '''
            delta_w2 = np.dot(hidden_layer_output.T, diff_y)
            delta_w1 = np.dot(X[[j], :].T, np.dot(diff_y, w2.T) * sigmoid_derivative_func(hidden_layer_output))

            w1 = w1 + (learning_rate * delta_w1)
            w2 = w2 + (learning_rate * delta_w2)
            '''

            #print(w1.grad,(w1.grad).size())
            w1.data.sub_(w1.grad.data * learning_rate)
            w2.data.sub_(w2.grad.data * learning_rate)

            #w1 = w1 + learning_rate*w1.grad
            #w2 = w2 + learning_rate*w2.grad
        pred = torch.argmax(y_output, 1)
        print(loss.data)
        #print(str(i) + ': ' + str(100 * torch.sum(pred == torch.squeeze(y_target)) / X.shape[0]))

        i += 1