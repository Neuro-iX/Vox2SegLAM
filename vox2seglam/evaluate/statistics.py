"""Paired comparison of two conditions measured on the same subjects."""

import numpy as np
from scipy.stats import wilcoxon

__all__ = ["paired_comparison"]


def paired_comparison(before, after, lower_is_better: bool = False) -> dict | None:
    """Means, relative change (%), two-sided Wilcoxon signed-rank p-value and the number of
    subjects improved; pairs with a nan are dropped, None when none is left."""
    a, b = np.asarray(before, float), np.asarray(after, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if not len(a):
        return None
    p = float(wilcoxon(b, a).pvalue) if np.any(a != b) else 1.0
    improved = (b < a) if lower_is_better else (b > a)
    rel = float(100 * (b.mean() - a.mean()) / a.mean()) if a.mean() else 0.0
    return {
        "before": float(a.mean()),
        "after": float(b.mean()),
        "relative_change": rel,
        "p": p,
        "improved": int(improved.sum()),
        "n": int(len(a)),
    }
