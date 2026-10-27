import numpy as np
from scipy.linalg import cho_factor, cho_solve


class IALS:
    """Weighted squared-error implicit ALS. Unobserved entries have p=0, c=1."""
    def __init__(self, factors=32, alpha=10, reg=1.0, iterations=15, seed=42):
        self.factors, self.alpha, self.reg = factors, alpha, reg
        self.iterations, self.seed = iterations, seed

    def _targets(self, counts, ages=None):
        p = counts.copy().astype(float)
        p.data[:] = 1.0
        return p

    def _step(self, confidence_delta, preference, fixed):
        result = np.zeros((confidence_delta.shape[0], self.factors))
        base = fixed.T @ fixed + self.reg * np.eye(self.factors)
        for row in range(confidence_delta.shape[0]):
            a, b = confidence_delta.indptr[row:row + 2]
            idx = confidence_delta.indices[a:b]
            if len(idx) == 0:
                continue
            v = fixed[idx]
            delta = confidence_delta.data[a:b]
            target = preference.data[a:b]
            lhs = base + (v.T * delta) @ v
            rhs = v.T @ ((1.0 + delta) * target)
            result[row] = cho_solve(cho_factor(lhs, check_finite=False), rhs, check_finite=False)
        return result

    def fit(self, counts, ages=None):
        if self.reg <= 0 or self.alpha < 0:
            raise ValueError("reg must be positive and alpha nonnegative")
        counts = counts.tocsr().astype(float)
        counts.sort_indices()
        preference = self._targets(counts, ages)
        delta = counts.copy()
        delta.data = self.alpha * np.log1p(delta.data)
        self.preference = preference
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.01, (counts.shape[1], self.factors))
        delta_t, p_t = delta.T.tocsr(), preference.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._step(delta, preference, self.item_factors)
            self.item_factors = self._step(delta_t, p_t, self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
