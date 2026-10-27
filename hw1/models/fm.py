"""Second-order FM, hand-written BPR gradients and Adam (NumPy/SciPy only)."""
import numpy as np
from scipy.special import expit


def fm_forward(x, w, v):
    xv = x @ v
    pairwise = 0.5 * (xv * xv - x.multiply(x) @ (v * v)).sum(axis=1)
    return np.asarray(x @ w).ravel() + pairwise


def bpr_loss_grad(xp, xn, w, v, reg=0):
    """Mean softplus(-[FM(x_pos)-FM(x_neg)]) plus L2/2."""
    margin = fm_forward(xp, w, v) - fm_forward(xn, w, v)
    a = -expit(-margin) / len(margin)
    xp_v, xn_v = xp @ v, xn @ v
    grad_w = np.asarray(xp.T @ a - xn.T @ a).ravel() + reg * w
    grad_v = np.asarray(xp.T @ (a[:, None] * xp_v) - xn.T @ (a[:, None] * xn_v))
    correction = np.asarray(xp.multiply(xp).T @ a - xn.multiply(xn).T @ a).ravel()
    grad_v -= correction[:, None] * v
    grad_v += reg * v
    loss = np.logaddexp(0, -margin).mean() + reg / 2 * (np.sum(w * w) + np.sum(v * v))
    return float(loss), grad_w, grad_v


def sample_negatives(users, known, rng):
    if np.any(np.diff(known.indptr)[users] >= known.shape[1]):
        raise ValueError("A user has no unseen items to sample")
    negatives = rng.integers(0, known.shape[1], len(users))
    bad = np.asarray(known[users, negatives]).ravel() > 0
    while bad.any():
        negatives[bad] = rng.integers(0, known.shape[1], int(bad.sum()))
        bad = np.asarray(known[users, negatives]).ravel() > 0
    return negatives


class FactorizationMachine:
    def __init__(self, factors=32, lr=0.003, reg=1e-5, epochs=12, batch_size=4096, seed=42):
        self.factors, self.lr, self.reg = factors, lr, reg
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed

    def fit(self, known, user_features, item_features):
        self.uf, self.itf = user_features, item_features
        if self.uf.shape[1] != self.itf.shape[1]:
            raise ValueError("Feature spaces must have the same dimension")
        # Fast scoring requires disjoint user-side and item-side feature namespaces.
        if np.intersect1d(self.uf.indices, self.itf.indices).size:
            raise ValueError("User and item feature namespaces must be disjoint")
        rng = np.random.default_rng(self.seed)
        dim = self.uf.shape[1]
        self.w = np.zeros(dim)
        self.v = rng.normal(0, 0.02, (dim, self.factors))
        moments = [np.zeros_like(p) for p in [self.w, self.v]]
        velocities = [np.zeros_like(p) for p in [self.w, self.v]]
        coo = known.tocoo()
        step, self.loss_history = 0, []
        for _ in range(self.epochs):
            order = rng.permutation(known.nnz)
            weighted_loss = 0.
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                u, pos = coo.row[batch], coo.col[batch]
                neg = sample_negatives(u, known, rng)
                xp, xn = self.uf[u] + self.itf[pos], self.uf[u] + self.itf[neg]
                loss, gw, gv = bpr_loss_grad(xp, xn, self.w, self.v, self.reg)
                weighted_loss += loss * len(batch)
                step += 1
                for param, grad, m, q in zip([self.w, self.v], [gw, gv], moments, velocities):
                    m *= 0.9; m += 0.1 * grad
                    q *= 0.999; q += 0.001 * grad * grad
                    param -= self.lr * (m / (1 - 0.9**step)) / (np.sqrt(q / (1 - 0.999**step)) + 1e-8)
            self.loss_history.append(weighted_loss / known.nnz)
        self.user_vectors, self.item_vectors = self.uf @ self.v, self.itf @ self.v
        self.user_bias = fm_forward(self.uf, self.w, self.v)
        self.item_bias = fm_forward(self.itf, self.w, self.v)
        return self

    def score(self, users):
        # FM(user + item) = FM(user) + FM(item) + <sum_user v*x, sum_item v*x>.
        return self.user_vectors[users] @ self.item_vectors.T + self.user_bias[users, None] + self.item_bias[None, :]
