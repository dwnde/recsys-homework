import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    """Exact solution of ||X-XB||_F^2 + lambda ||B||_F^2, diag(B)=0."""
    def __init__(self, reg=200):
        self.reg = reg

    def fit(self, x):
        if self.reg <= 0:
            raise ValueError("reg must be positive")
        self.x = x.tocsr().astype(float)
        gram = (self.x.T @ self.x).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        precision = cho_solve(cho_factor(gram, check_finite=False), np.eye(gram.shape[0]), check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.x[users] @ self.weights
