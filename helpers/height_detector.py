"""Ball height on every frame that has an ATD ball position (stage 2).

The ATD ball is the ball's position projected on the ground. The real ball lies on the vertical line above it, which in
the image is a (nearly straight) curve. Per frame we take the candidates within TOL_PX of that curve, score them with the
classifier, and pick the best sequence through time with a Viterbi pass. The height is read off the curve at the point
closest to the chosen candidate. The match is processed in blocks of consecutive frames, in parallel, and the blocks are
merged into one table.

Not used, on purpose: widening TOL_PX beyond 30. Wider bands admit boots and bodies; the failures that remain (ball
several metres from the ATD point, or ball hidden in the net) cannot be fixed that way.
"""
import glob
import os

import numpy as np
import pandas as pd

from .ball_classifier import BallClassifier
from .camera import Camera
from .frame_reader import FrameReader
from .match import Match
from .parallel import ParallelJobs
from .settings import Settings


class HeightDetector:
    Z_GRID = np.arange(-0.5, 8.01, 0.05)   # heights tried along the vertical line (metres)
    TOL_PX = 30.0                          # max distance of a candidate from the curve
    SIGMA_PX = 10.0                        # distance penalty scale
    Z_STEP_M = 0.8                         # height-continuity scale between consecutive samples
    MISS_LOGP = np.log(0.08)               # cost of "ball not found in this frame"
    BLOCK_SIZE = 100
    BLOCK_OVERLAP = 10
    COLUMNS = ["frame", "detected", "x", "y", "z", "res", "p", "px", "py"]

    def __init__(self, match, classifier=None):
        self.m = match
        self.clf = classifier
        self.reader = FrameReader(match)

    # ---- detection on a run of consecutive frames ------------------------------------------------------------------
    def vertical_curve(self, f):
        """Image points of the vertical line above the ATD ball position of frame f (len(Z_GRID), 2)."""
        cam = Camera(self.m.H[f])
        return cam.project(np.array([[self.m.bx[f], self.m.by[f], z] for z in self.Z_GRID]))

    def detect(self, frames, cands, probs, tol=None, sig=None):
        """Viterbi over a run of consecutive samples. Returns, per frame, None or a dict
        (c: candidate, z: height m, res: px distance to the curve, p: classifier score)."""
        tol = self.TOL_PX if tol is None else tol
        sig = self.SIGMA_PX if sig is None else sig
        m = self.m
        T = len(frames)
        opts = []
        for fr, cs, pr in zip(frames, cands, probs):
            o = []
            if np.isfinite(m.bx[fr]) and cs:
                cv = self.vertical_curve(fr)
                for c, p in zip(cs, pr):
                    dd = np.hypot(cv[:, 0] - c["x"], cv[:, 1] - c["y"])
                    k = int(np.argmin(dd))
                    if dd[k] < tol:
                        o.append(dict(c=c, z=float(self.Z_GRID[k]), res=float(dd[k]), p=float(p),
                                      sc=float(np.log(p + 1e-3) - 0.5 * (dd[k] / sig) ** 2)))
            opts.append(o)
        nodes = [[None] + list(range(len(o))) for o in opts]
        dp = [np.full(len(n), -1e9) for n in nodes]
        bp = [np.zeros(len(n), int) for n in nodes]
        for j, nd in enumerate(nodes[0]):
            dp[0][j] = self.MISS_LOGP if nd is None else opts[0][nd]["sc"]
        for t in range(1, T):
            zprev = np.array([np.nan if pd_ is None else opts[t - 1][pd_]["z"] for pd_ in nodes[t - 1]])
            for j, nd in enumerate(nodes[t]):
                em = self.MISS_LOGP if nd is None else opts[t][nd]["sc"]
                if nd is None:
                    tr = np.full(len(zprev), -0.3)
                else:
                    tr = np.where(np.isnan(zprev), -0.3,
                                  -np.minimum(0.5 * ((opts[t][nd]["z"] - np.nan_to_num(zprev)) / self.Z_STEP_M) ** 2, 4))
                tot = dp[t - 1] + tr
                arg = int(np.argmax(tot))
                dp[t][j] = tot[arg] + em
                bp[t][j] = arg
        j = int(np.argmax(dp[-1]))
        path = [None] * T
        for t in range(T - 1, -1, -1):
            nd = nodes[t][j]
            path[t] = None if nd is None else opts[t][nd]
            j = bp[t][j]
        return path

    # ---- blocks ----------------------------------------------------------------------------------------------------
    def block_starts(self):
        """Start frames of the blocks (consecutive BLOCK_SIZE frames inside the play ranges) that contain at least
        one ATD ball position. Blocks with none need no processing."""
        out = []
        for a, b in Settings.PLAY_RANGES:
            for s in range(a, b, self.BLOCK_SIZE):
                if self.m.has_ball[s:min(s + self.BLOCK_SIZE, b)].any():
                    out.append(s)
        return out

    def process_block(self, start):
        """Run the detector on frames [start, start + BLOCK_SIZE). BLOCK_OVERLAP extra frames are read after the block
        so the temporal smoothing is not cut off at the block edge; they are not returned.
        Returns an array (n, 9) with COLUMNS: frame, detected (0/1), pixel x/y of the chosen candidate, height z in
        metres above the ground, px distance to the curve, classifier score, and the candidate's ground-plane
        position in metres (px, py)."""
        end = Settings.range_of(start)[1] if Settings.range_of(start) else start
        count = min(self.BLOCK_SIZE + self.BLOCK_OVERLAP, end - start)
        frames, cands, _ = self.reader.read_window(start, count, step=1)
        path = self.detect(frames, cands, [self.clf.proba(cs) for cs in cands])
        rows = []
        for fr, p in zip(frames, path):
            if fr >= start + self.BLOCK_SIZE:
                break
            if p is None:
                rows.append([fr, 0] + [np.nan] * 7)
            else:
                c = p["c"]
                rows.append([fr, 1, c["x"], c["y"], p["z"], p["res"], p["p"], c["px"], c["py"]])
        return np.array(rows, np.float64)

    # ---- the whole match, in parallel (one file per block in outputs/blocks/, so a run can be resumed) --------------
    @staticmethod
    def _make_context():
        return HeightDetector(Match(), BallClassifier.load(Settings.out("classifier.joblib")))

    @staticmethod
    def _work(detector, start):
        path = Settings.out("blocks", "b%06d.npy" % start)
        if os.path.exists(path):
            return start
        np.save(path + ".tmp.npy", detector.process_block(start))
        os.replace(path + ".tmp.npy", path)
        return start

    @classmethod
    def run_blocks(cls, workers=6, limit=None, only_start=None, log=print):
        os.makedirs(Settings.out("blocks"), exist_ok=True)
        starts = cls(Match()).block_starts()
        if only_start is not None:
            starts = [only_start]
        if limit:
            starts = starts[:limit]
        log("%d blocks to process" % len(starts))
        ParallelJobs(cls._make_context, cls._work, workers, log).run(starts)

    @classmethod
    def aggregate(cls, log=print):
        """outputs/blocks/*.npy -> outputs/ball_height.csv (one row per frame with an ATD ball position) and
        outputs/block_stats.csv."""
        files = sorted(glob.glob(Settings.out("blocks", "b*.npy")))
        d = pd.DataFrame(np.vstack([np.load(f) for f in files]), columns=cls.COLUMNS)
        d["frame"] = d["frame"].astype(int)
        d["detected"] = d["detected"].astype(int)
        d = d[Match().has_ball[d["frame"].values]]
        d.to_csv(Settings.out("ball_height.csv"), index=False)
        d["block"] = (d["frame"] // 100) * 100
        st = d.groupby("block").agg(atd_frames=("frame", "size"), detected=("detected", "sum"))
        st["coverage"] = st["detected"] / st["atd_frames"]
        st.to_csv(Settings.out("block_stats.csv"))
        log("blocks: %d | ATD frames: %d | detected: %d (%.1f%%)" % (len(st), len(d), d.detected.sum(), 100 * d.detected.mean()))
        log("coverage per block: median %.0f%% | p10 %.0f%% | p90 %.0f%% | blocks under 20%%: %d" % (
            100 * st.coverage.median(), 100 * st.coverage.quantile(.1), 100 * st.coverage.quantile(.9), (st.coverage < .2).sum()))
        z = d.loc[d.detected == 1, "z"]
        log("height of detections: <0.3 m %.0f%% | 0.3-1 m %.0f%% | 1-2 m %.0f%% | >2 m %.0f%%" % (
            100 * (z < .3).mean(), 100 * ((z >= .3) & (z < 1)).mean(), 100 * ((z >= 1) & (z < 2)).mean(), 100 * (z >= 2).mean()))
        return d
