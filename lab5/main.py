import RBFNet as rBFNet
import RNN as rNN

import numpy as np
import matplotlib.pyplot as plt

import NeuralNetwork as nn

def TraningFunc(fx, rbf, perceptron_1, perceptron_2, perceptron_1x2, perceptron_1x3):
    x = np.random.uniform(-4.5, 4.5, size=[75000, 1])
    y = fx(x)
    rbf.sigma = np.std(y)
    rbf.fit(x, y)
    for i in range(75000):
        xi = x[i].reshape(1, 1)
        yi = y[i].reshape(1, 1)
        perceptron_1.forward(xi)
        perceptron_1.backward(yi)
        perceptron_1.update()
        perceptron_2.forward(xi)
        perceptron_2.backward(yi)
        perceptron_2.update()
        perceptron_1x2.forward(xi)
        perceptron_1x2.backward(yi)
        perceptron_1x2.update()
        perceptron_1x3.forward(xi)
        perceptron_1x3.backward(yi)
        perceptron_1x3.update()
    return True


def TraningFunc2(seq_length, train, rnn, perceptron_3, mean, std):
    for epoch in range(500):
        q = np.random.randint(0, seq_length)
        for i in range(q, len(train) - seq_length, seq_length):
            x = (np.array(train[i: i + seq_length]).reshape(1, seq_length) - mean) / std
            y = (np.array(train[i + seq_length: i + seq_length + 1]).reshape(1, 1) - mean) / std
            rnn(x)
            rnn.backward(y)
            rnn.update()
            perceptron_3.forward(x)
            perceptron_3.backward(y)
            perceptron_3.update()
    return True


def create_nets_for_task_1():
    rbf = rBFNet.RBFNet(10)
    perceptron_1 = nn.NN(0.0001)
    perceptron_1.add_layer(1, 6, "tanh", need_bias=True)
    perceptron_1.add_layer(6, 1)
    perceptron_2 = nn.NN(0.0001)
    perceptron_2.add_layer(1, 6, "tanh", need_bias=True)
    perceptron_2.add_layer(6, 6, "tanh", need_bias=True)
    perceptron_2.add_layer(6, 1)
    perceptron_1x2 = nn.NN(0.0001)
    perceptron_1x2.add_layer(1, 8, "tanh", need_bias=True)
    perceptron_1x2.add_layer(8, 1)
    perceptron_1x3 = nn.NN(0.0001)
    perceptron_1x3.add_layer(1, 30, "tanh", need_bias=True)
    perceptron_1x3.add_layer(30, 1)
    return rbf, perceptron_1, perceptron_2, perceptron_1x2, perceptron_1x3


def create_nets_for_task_2(seq_length):
    rnn = rNN.RNN(seq_length, 40, 1, 0.003)
    perceptron_3 = nn.NN(0.003)
    perceptron_3.add_layer(seq_length, 40, "tanh")
    perceptron_3.add_layer(40, 1)
    return rnn, perceptron_3


def deserialize(filename):
    with open(filename, "r") as file:
        rows = file.read().split()
        data = []
        for i, k in enumerate(rows[:-1]):
            data.append(float(k))
    return data


if __name__ == "__main__":
    train = deserialize(r"train.txt")
    test = deserialize(r"test.txt")
    mean = np.mean(train)
    std = np.std(train)

    training_complete = False
    fx = lambda x: x * x * np.exp(np.sin(x))
    rbf, perceptron_1, perceptron_2, perceptron_1x2, perceptron_1x3 = create_nets_for_task_1()

    training_complete_2 = False
    seq_length = 30
    rnn, perceptron_3 = create_nets_for_task_2(seq_length)

    flag = False
    k = 0
    training_complete = TraningFunc(fx, rbf, perceptron_1, perceptron_2, perceptron_1x2, perceptron_1x3)
    training_complete_2 = TraningFunc2(seq_length, train, rnn, perceptron_3, mean, std)
    while not flag:
        print(
            "1. График аппроксимации\n2. Сравнение сетей с разным кол-вом нейронов на скрытом слое\n3. График температуры\n4. График предсказаний на январь\n")
        try:
            k = int(input())
        except ValueError:
            pass
        print()

        if k == 1:
            if training_complete:
                points = np.linspace(-4.5, 4.5, 400)
                p1 = []
                p2 = []
                for i in points:
                    p1.append(perceptron_1(i.reshape(1, 1)).reshape(-1))
                    p2.append(perceptron_2(i.reshape(1, 1)).reshape(-1))
                plt.plot(points, rbf.predict(points), "r", label="Радиально-базисная сеть")
                plt.plot(points, p2, "g", label="1 скрытый слой")
                plt.plot(points, p1, "b", label="2 скрытых слоя")
                plt.plot(points, fx(points), "k")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.grid()
                plt.show()
        elif k == 2:
            if training_complete:
                points = np.linspace(-4.5, 4.5, 400)
                p1 = []
                p1_x2 = []
                p1_x3 = []
                for i in points:
                    p1.append(perceptron_1(i.reshape(1, 1)).reshape(-1))
                    p1_x2.append(perceptron_1x2(i.reshape(1, 1)).reshape(-1))
                    p1_x3.append(perceptron_1x3(i.reshape(1, 1)).reshape(-1))
                plt.plot(points, p1, "r", label="6 н")
                plt.plot(points, p1_x2, "g", label="8 н")
                plt.plot(points, p1_x3, "b", label="30 н")
                plt.plot(points, fx(points), "k")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.grid()
                plt.show()
        elif k == 3:
            plt.plot(train)
            plt.grid()
            plt.show()
        elif k == 4:
            if training_complete_2:
                rnn_y = []
                perceptron_y = []
                for i in train[-seq_length:]:
                    rnn_y.append((i - mean) / std)
                    perceptron_y.append((i - mean) / std)
                for i in range(len(test)):
                    out_rnn = rnn(np.array(rnn_y[i:i + seq_length]).reshape(1, seq_length)).reshape(-1)
                    out_perceptron_3 = perceptron_3(
                        np.array(perceptron_y[i:i + seq_length]).reshape(1, seq_length)).reshape(-1)
                    rnn_y.append(out_rnn[0])
                    perceptron_y.append(out_perceptron_3[0])
                plt.plot(np.array(rnn_y[seq_length:]) * std + mean, label="Сеть Элмана")
                plt.plot(np.array(perceptron_y[seq_length:]) * std + mean, label="Персептрон")
                plt.plot(test, label="Курс бел.р. к юаню")
                plt.legend()
                plt.grid()
                plt.show()
