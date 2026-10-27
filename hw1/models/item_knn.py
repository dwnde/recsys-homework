import numpy as np
from scipy import sparse


class ItemKNN:
    """Cosine item neighbors with co-occurrence shrinkage, binary histories."""
    def __init__(self, neighbors=100, shrink=10):
        self.neighbors, self.shrink = neighbors, shrink

    def fit(self, x):
        self.x = x.tocsr().astype(float)
        gram = (self.x.T @ self.x).toarray()
        norms = np.sqrt(np.diag(gram))
        sim = gram / np.maximum(norms[:, None] * norms[None, :], 1e-12)
        sim *= gram / np.maximum(gram + self.shrink, 1e-12)
        np.fill_diagonal(sim, 0)
        # Each candidate column retains its strongest source-item neighbors.
        rows, cols, vals = [], [], []
        for j in range(sim.shape[1]):
            selected = np.argsort(-sim[:, j], kind="stable")[:self.neighbors]
            selected = selected[sim[selected, j] > 0]
            rows.extend(selected)
            cols.extend([j] * len(selected))
            vals.extend(sim[selected, j])
        self.weights = sparse.csc_matrix((vals, (rows, cols)), shape=sim.shape)
        return self

    def score(self, users):
        return (self.x[users] @ self.weights).toarray()
