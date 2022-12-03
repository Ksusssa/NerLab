import numpy as np


class RBFNet:
    def __init__(self, hidden_number, sigma=1.0):
        self.hidden_number = hidden_number
        self.sigma = sigma
        self.centers = 0
        self.weights = 0

    def rbf(self, point, center):
        return np.exp(np.linalg.norm((point - center) ** 2 / (2 * self.sigma ** 2)))

    def calculate_interpolation_matrix(self, x):
        g = np.zeros((len(x), self.hidden_number))
        for i, point in enumerate(x):
            for j, center in enumerate(self.centers):
                g[i, j] = self.rbf(point, center)
        return g

    def fit(self, x, y):
        self.centers = x[np.random.choice(len(x), self.hidden_number)]
        g = self.calculate_interpolation_matrix(x)
        inv_g = np.linalg.pinv(g)
        self.weights = inv_g @ y

    def predict(self, x):
        g = self.calculate_interpolation_matrix(x)
        return g @ self.weights

