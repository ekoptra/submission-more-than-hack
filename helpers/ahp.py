"""Analytic Hierarchy Process (Saaty): criteria weights from pairwise comparisons, with a consistency check, and the
pairwise judgements of docs/technical_document.md, section 6.

The judgements are written down here so they can be inspected. They come from football reasoning (reliability, coverage
and decisiveness of each criterion), not from any data or ground truth. In each upper-triangle entry (i, j), the value
says how many times more important criterion i is than criterion j (below 1 means less important).
"""
import numpy as np
import pandas as pd


class AHP:
    RANDOM_INDEX = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}

    DEAD_BALL = (
        ["D1 pemain pelan", "D2 bola di luar garis", "D3 pemain berhenti", "D4 bola diam", "D5 bola hilang",
         "D6 Smart Tagging", "D7 pemain di titik restart"],
        {(0, 1): 2, (0, 2): 3, (0, 3): 5, (0, 4): 7, (0, 5): 4, (1, 2): 2, (1, 3): 3, (1, 4): 5, (1, 5): 3,
         (2, 3): 2, (2, 4): 3, (2, 5): 2, (3, 4): 2, (3, 5): 1 / 2, (4, 5): 1 / 3,
         (0, 6): 4, (1, 6): 3, (2, 6): 2, (3, 6): 1 / 2, (4, 6): 1 / 3, (5, 6): 1})
    TEAM_BALL_KNOWN = (
        ["T1 kedekatan", "T2 dominasi lokal", "T3 penguasa terakhir", "T4 Smart Tagging", "T5 bentuk tim",
         "T6 kesesuaian gerak"],
        {(0, 1): 3, (0, 2): 3, (0, 3): 7, (0, 4): 5, (0, 5): 2,
         (1, 2): 1, (1, 3): 4, (1, 4): 3, (1, 5): 1 / 2,
         (2, 3): 4, (2, 4): 3, (2, 5): 1 / 2,
         (3, 4): 1 / 2, (3, 5): 1 / 5, (4, 5): 1 / 4})
    TEAM_BALL_UNKNOWN = (
        ["T3 penguasa terakhir", "T5 bentuk tim", "T4 Smart Tagging"],
        {(0, 1): 1, (0, 2): 4, (1, 2): 4})
    DEAD_BALL_NOT_USED = ("D4",)      # the ball moves during a stoppage, so "ball almost still" is dropped

    def __init__(self, names, upper):
        """names: criterion names. upper: {(i, j): importance of i over j} for i < j (Saaty scale 1-9)."""
        self.names = list(names)
        n = len(self.names)
        self.matrix = np.ones((n, n))
        for (i, j), v in upper.items():
            self.matrix[i, j] = v
            self.matrix[j, i] = 1.0 / v
        values, vectors = np.linalg.eig(self.matrix)
        k = int(np.argmax(values.real))
        w = np.abs(vectors[:, k].real)
        self._weights = w / w.sum()
        self.lambda_max = float(values[k].real)
        self.ci = (self.lambda_max - n) / (n - 1)
        ri = self.RANDOM_INDEX[n]
        self.cr = self.ci / ri if ri else 0.0

    def weights(self):
        """{criterion code: weight}, the code being the first word of the name (for example 'D1')."""
        return {name.split()[0]: float(w) for name, w in zip(self.names, self._weights)}

    def without(self, codes):
        """The same judgements with some criteria removed (rows and columns dropped, weights recomputed)."""
        keep = [i for i, name in enumerate(self.names) if name.split()[0] not in codes]
        upper = {(a, b): self.matrix[keep[a], keep[b]] for a in range(len(keep)) for b in range(a + 1, len(keep))}
        return AHP([self.names[i] for i in keep], upper)

    @classmethod
    def dead_ball_weights(cls):
        """Weights actually used for the dead-ball criteria (D4 removed, the others renormalised)."""
        return cls(*cls.DEAD_BALL).without(cls.DEAD_BALL_NOT_USED).weights()

    @classmethod
    def team_weights(cls):
        """(weights when the ball is known, weights when it is not)."""
        return cls(*cls.TEAM_BALL_KNOWN).weights(), cls(*cls.TEAM_BALL_UNKNOWN).weights()

    # ---- combining the criteria with the weights (docs/technical_document.md, section 5.3) ----------------------------------
    #   s_D = sum(w_k D_k) / sum(w_k)                over the D criteria computable on the frame
    #   q_H = sum(w_k a_k T_k) / sum(w_k a_k)        adaptive weights: a = c for T1 and T2, e for T6, 1 for the others
    #   P(D) = s_D,  P(H) = (1 - s_D) q_H,  P(A) = (1 - s_D) (1 - q_H)
    # The set of T weights depends on whether the ball is known on the frame (T1 computable) or not.
    @classmethod
    def dead_ball_score(cls, criteria):
        num = np.zeros(len(criteria))
        den = np.zeros(len(criteria))
        for code, w in cls.dead_ball_weights().items():
            v = criteria[code].values
            ok = np.isfinite(v)
            num += np.where(ok, w * v, 0)
            den += np.where(ok, w, 0)
        return num / den

    @classmethod
    def home_share(cls, criteria):
        n = len(criteria)
        known = criteria["T1"].notna().values
        adaptive = {"T1": criteria["c"].values, "T2": criteria["c"].values, "T6": criteria["e"].values}
        w_known, w_unknown = cls.team_weights()
        num, den = np.zeros(n), np.zeros(n)
        for weights, use, use_adaptive in ((w_known, known, True), (w_unknown, ~known, False)):
            for code, w in weights.items():
                v = criteria[code].values
                ok = np.isfinite(v) & use
                a = adaptive.get(code, np.ones(n)) if use_adaptive else np.ones(n)
                a = np.where(np.isfinite(a), a, 0.0)
                num += np.where(ok, w * a * v, 0)
                den += np.where(ok, w * a, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, 0.5)

    @classmethod
    def probabilities(cls, criteria):
        """criteria (DataFrame from outputs/criteria.csv.gz) -> DataFrame: frame, sD, qH, P_H, P_A, P_D."""
        sD, qH = cls.dead_ball_score(criteria), cls.home_share(criteria)
        return pd.DataFrame({"frame": criteria.frame.values, "sD": sD, "qH": qH, "P_H": (1 - sD) * qH,
                             "P_A": (1 - sD) * (1 - qH), "P_D": sD})
