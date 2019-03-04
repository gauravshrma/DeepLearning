import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np


class Network(nn.Module):
    def __init__(self,num_feats,num_out,num_hid):
        super(Network, self).__init__()

        self.inputs = num_feats
        self.hidden_units = num_hid
        self.outputs = num_out

        self.layer_1_linear = nn.Linear(self.inputs, self.hidden_units)

        self.sigmoid = nn.Sigmoid()

        self.layer_2_linear = nn.Linear(self.hidden_units,self.outputs)

    def forward(self, X):
        linear_1 = self.layer_1_linear(X)
        hidden_layer_output = self.sigmoid(linear_1)
        y_output = self.layer_2_linear(hidden_layer_output)
        return y_output


if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris.data[:, :]  # we only take the first two features.
    row_X = X.shape[0]
    num_feats = X.shape[1]

    for j in range(num_feats):
        X[:, j] = (X[:, j] - np.min(X[:, j])) / (max(X[:, j]) - np.min(X[:, j], 0))

    y = iris.target
    y_target = np.zeros((y.shape[0], 1))
    y_target[:, 0] = y

    rand_seq = np.random.permutation(row_X)
    X = X[rand_seq, :]
    y_target = y_target[rand_seq, :]

    y_target_one_hot = np.zeros((row_X, 3))
    for i in range(row_X):
        if (y_target[i, 0] == 0):
            y_target_one_hot[i, 0] = 1
        elif (y_target[i, 0] == 1):
            y_target_one_hot[i, 1] = 1
        else:
            y_target_one_hot[i, 2] = 1

    # parameters for forward Propogation



    # convert numpy arrays to torch
    X = (torch.from_numpy(X)).float()
    y_target = torch.from_numpy(y_target)
    y_target_one_hot = (torch.from_numpy(y_target_one_hot)).float()
    y_output = (torch.tensor(np.zeros(y_target_one_hot.shape))).float()


    model = Network(num_feats,num_out=y_target_one_hot.shape[1],num_hid=20)

    # hyperparameters
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    i = 0

    criterion = nn.MSELoss()
    while i < 1000:
        for j in range(X.shape[0]):

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            output = model(X[j,:])
            y_output[j,:] = output

            loss = criterion(output, y_target_one_hot[j,:])

            loss.backward()

            # Updating parameters
            optimizer.step()

        pred = torch.argmax(y_output, 1)
        print(loss.data)
        # print(str(i) + ': ' + str(100 * torch.sum(pred == torch.squeeze(y_target)) / X.shape[0]))

        i += 1