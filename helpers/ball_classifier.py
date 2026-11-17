"""Gradient-boosting classifier that scores a ball candidate from its 13 features, and its training from the ATD itself
(no hand labels)."""
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .camera import Camera
from .frame_reader import FrameReader


class BallClassifier:
    # Windows that are kept out of training (they are used for evaluation).
    EVAL_STARTS = [24480, 41040, 66520, 81840, 117600, 121440, 134640, 151240, 20100, 170130]
    WINDOW = 100

    def __init__(self, model=None):
        self.model = model

    @classmethod
    def fit(cls, Xs, ys):
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced",
                                               random_state=0).fit(np.vstack(Xs), np.concatenate(ys))
        return cls(model)

    def proba(self, candidates):
        """Probability that each candidate is the ball (empty list for no candidates)."""
        return self.model.predict_proba(np.array([c["fv"] for c in candidates]))[:, 1] if candidates else []

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))

    @staticmethod
    def ground_labels(match, frames, cands):
        """Training labels from the ATD ground position: positive = candidate within 10 px of the ATD ground
        pixel (a ball on the grass); negative = more than 40 px away and outside the band above it where an
        airborne ball could be (25 px wide, up to 130 px above). Candidates in between are left unlabelled."""
        xs, ys = [], []
        for fr, cs in zip(frames, cands):
            if not np.isfinite(match.bx[fr]):
                continue
            kx, ky = Camera.to_pixel(match.H[fr], match.B[fr, 0], match.B[fr, 1])
            for c in cs:
                dd = np.hypot(c["x"] - kx, c["y"] - ky)
                band = abs(c["x"] - kx) < 25 and -130 <= c["y"] - ky <= 0
                if dd < 10:
                    xs.append(c["fv"])
                    ys.append(1)
                elif dd > 40 and not band:
                    xs.append(c["fv"])
                    ys.append(0)
        return np.array(xs), np.array(ys)

    @classmethod
    def select_windows(cls, match, n=24, seed=11):
        """Deterministic list of 100-frame windows where the ATD ball is present on >= 95% of frames, away from the
        evaluation windows. The first 12 train the classifier."""
        rng = np.random.default_rng(seed)
        out = []
        for f0 in rng.permutation(np.arange(12000, 189000, 50)):
            if (match.has_ball[f0:f0 + cls.WINDOW].mean() >= 0.95
                    and all(abs(f0 - e) > 1500 for e in cls.EVAL_STARTS)
                    and all(abs(f0 - q) > 600 for q in out) and len(out) < n):
                out.append(int(f0))
        return out

    @classmethod
    def train(cls, match, log=print):
        """Stage 1: positives = candidates on the ATD ground pixel (balls on the grass).
        Stage 2: run the height detector with that classifier on the same training windows and add its detections
        (within 8 px of the curve) as extra positives, which is how airborne balls enter the training set.
        Negatives in stage 2 are candidates more than 40 px from the vertical curve and not chosen."""
        from .height_detector import HeightDetector          # imported here: the detector itself uses this class
        reader, detector = FrameReader(match), HeightDetector(match)
        trn = []
        for f0 in cls.select_windows(match)[:12]:
            fr, cs, _ = reader.read_window(f0, cls.WINDOW, step=2)
            trn.append((fr, cs))
            log("training window %d read" % f0)
        Xg, yg = zip(*[cls.ground_labels(match, fr, cs) for fr, cs in trn])
        clf1 = cls.fit(Xg, yg)
        log("stage 1: %d candidates, %d positives" % (sum(len(a) for a in yg), sum(a.sum() for a in yg)))
        Xp, yp = [], []
        for fr_, cs_ in trn:
            path = detector.detect(fr_, cs_, [clf1.proba(c) for c in cs_])
            for fr, cs, p in zip(fr_, cs_, path):
                if not np.isfinite(match.bx[fr]):
                    continue
                cv = detector.vertical_curve(fr)
                for c in cs:
                    dmin = float(np.min(np.hypot(cv[:, 0] - c["x"], cv[:, 1] - c["y"])))
                    if p is not None and c is p["c"] and p["res"] < 8:
                        Xp.append(c["fv"])
                        yp.append(1)
                    elif dmin > 40 and (p is None or c is not p["c"]):
                        Xp.append(c["fv"])
                        yp.append(0)
        Xp, yp = np.array(Xp), np.array(yp)
        clf = cls.fit(list(Xg) + [Xp], list(yg) + [yp])
        log("stage 2: %d candidates, %d positives (%d added from detections)"
            % (sum(len(a) for a in yg) + len(yp), sum(a.sum() for a in yg) + yp.sum(), yp.sum()))
        return clf
