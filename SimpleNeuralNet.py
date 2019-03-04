import numpy as np
from sklearn import datasets


def sigmoid_func(inputs):
    return (1 / (1 + np.exp(-inputs)))


def sigmoid_derivative_func(inputs):
    return (np.exp(-inputs) / (1 + np.exp(-inputs)) ** 2)


def linear_derivative_func(inputs):
    return (np.ones((inputs.shape[0], inputs.shape[1])))


def linear_func(inputs):
    return (inputs)

if __name__ == '__main__':

    learning_rate = 0.001

    iris = datasets.load_iris()
    X = iris.data[:, :]  # we only take the first two features.

    for j in range(X.shape[1]):
        X[:, j] = (X[:, j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))

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
    w1 = np.random.randn(inputs, hidden_units)
    w2 = np.random.randn(hidden_units, outputs)
    y_output = np.zeros(y_target_one_hot.shape)

    i = 0
    while i < 1000:
        for j in range(X.shape[0]):
            hidden_layer_output = sigmoid_func(np.dot(X[[j], :], w1))
            net_output = np.dot(hidden_layer_output, w2)
            y_output[j, :] = net_output

            diff_y = y_target_one_hot[j, :] - net_output

            delta_w2 = np.dot(hidden_layer_output.T, diff_y)
            delta_w1 = np.dot(X[[j], :].T, np.dot(diff_y, w2.T) * sigmoid_derivative_func(hidden_layer_output))

            w1 = w1 + (learning_rate * delta_w1)
            w2 = w2 + (learning_rate * delta_w2)

        pred = np.argmax(y_output, axis=1)
        print(str(i) + ': ' + str(100 * np.sum(pred == np.squeeze(y_target)) / X.shape[0]))

        i += 1