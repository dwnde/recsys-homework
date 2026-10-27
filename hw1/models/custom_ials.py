import numpy as np
from .ials import IALS


class CustomIALS(IALS):
    """Change preference p itself; confidence is identical to baseline iALS.

    p_ui = (1-rho) + rho * (1-exp(-n_ui)) * 2**(-age_ui/half_life)
    on observed entries, and zero elsewhere. n_ui counts distinct invoices.
    """
    def __init__(self, rho=0.5, half_life=90, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= rho < 1 or half_life <= 0:
            raise ValueError("rho must be in [0,1), half_life positive")
        self.rho, self.half_life = rho, half_life

    def _targets(self, counts, ages=None):
        if ages is None:
            raise ValueError("Past-event ages are required for the custom target")
        ages = ages.tocsr(copy=True)
        ages.sort_indices()
        if not (np.array_equal(ages.indptr, counts.indptr) and np.array_equal(ages.indices, counts.indices)):
            raise ValueError("Age and count matrices must have identical stored support")
        p = counts.copy().astype(float)
        p.data = (1 - self.rho) + self.rho * (-np.expm1(-counts.data)) * np.exp2(-ages.data / self.half_life)
        return p
