"""Small numeric helpers shared by the criteria."""
import numpy as np


class Scoring:
    @staticmethod
    def ramp(x, a, b):
        """clip((x - a) / (b - a), 0, 1): 0 at x = a, 1 at x = b, linear in between (decreasing when b < a).
        NaN stays NaN."""
        return np.clip((x - a) / (b - a), 0.0, 1.0)

    @staticmethod
    def window_mean(a, half):
        """Mean over +-half frames of the finite values of `a` (NaN where the window has no finite value)."""
        ok = np.isfinite(a)
        cs = np.concatenate(([0.0], np.cumsum(np.where(ok, a, 0.0))))
        cn = np.concatenate(([0], np.cumsum(ok)))
        i = np.arange(len(a))
        lo = np.maximum(0, i - half)
        hi = np.minimum(len(a), i + half + 1)
        n = cn[hi] - cn[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)
