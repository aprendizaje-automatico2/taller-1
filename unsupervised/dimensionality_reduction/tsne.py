import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit_transform(self, X):
        # Inicialización
        if self.random_state is not None:
            np.random.seed(self.random_state)
        N, D = X.shape
        Y = np.random.normal(0.0, 1e-4, size=(N, self.n_components))
        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)
        gains = np.ones_like(Y)

        # Preprocesamiento de los datos
        P = self.compute_joint_probabilities(X, self.perplexity)
        P = P + np.transpose(P)
        P = P / (2.0 * N)
        P = np.maximum(P, 1e-12)

        # Optimización
        for i in range(self.n_iter):
            # Calcula las distancias euclidianas en el espacio Y
            D = pdist(Y, "sqeuclidean")
            D = squareform(D)

            # Calcula las probabilidades condicionales en el espacio Y
            Q = self.compute_conditional_probabilities(D)
            Q = np.maximum(Q, 1e-12)

            # Calcula el gradiente
            PQ = P - Q
            for j in range(N):
                dY[j,:] = np.sum(np.tile(PQ[:,j] * self.gradient(D[j,:], Y[j,:]), (self.n_components, 1)).T, axis=0)

            # Actualiza la posición de los puntos
            gains = (gains + 0.2) * ((dY > 0.0) != (iY > 0.0)) + (gains * 0.8) * ((dY > 0.0) == (iY > 0.0))
            gains[gains < 0.01] = 0.01
            iY = self.learning_rate * (gains * dY - (self.early_exaggeration * gains * PQ.dot(Q)))
            Y = Y + iY

            # Normaliza la posición de los puntos
            Y = Y - np.mean(Y, axis=0)

        return Y

    def compute_joint_probabilities(self, X, perplexity):
        # Calcula la matriz de distancias euclidianas en el espacio original
        D = pdist(X
