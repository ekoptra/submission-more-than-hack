"""Stages 3-4: fill the stretches where the ATD has no ball, and build one ball position per graded frame.

A tracker follows the ball from a known image position, forward from the last detection before a gap and backward from
the first detection after it. Gaps of at most 5 s are interpolated instead. The result is one ball ground position per
graded frame, with the source of each value:
  atd       the ATD ball position (the ball's shadow on the pitch)
  interp    short gap: straight line between the ATD positions at both gap ends
  agree     longer gap: forward and backward trackers agree (< 5 m), average of the two
  one       longer gap: only one tracker answered, its ground position
  none      longer gap with no usable answer (no tracker answer, or the two trackers disagree)
"""
import glob
import os

import numpy as np
import pandas as pd

from .ball_classifier import BallClassifier
from .ball_tracker import BallTracker
from .frame_reader import FrameReader
from .match import Match
from .parallel import ParallelJobs
from .settings import Settings


class GapFiller:
    CHUNK = 64
    SEED_SEARCH = 125          # how far (frames) from a gap end we look for the last/first detection
    SHORT_GAP = 125            # gaps up to this length (5 s) are interpolated
    AGREE = 5.0                # forward and backward tracks agree when closer than this (metres)
    OUT_COLUMNS = ["frame", "f_ok", "f_gx", "f_gy", "f_x", "f_y", "b_ok", "b_gx", "b_gy", "b_x", "b_y"]

    def __init__(self, match, classifier, detections, params=None):
        """detections: {frame: (pixel x, pixel y, height z)} for frames where the height detector found the ball."""
        self.m, self.clf, self.det, self.params = match, classifier, detections, params
        self.reader = FrameReader(match)

    @staticmethod
    def find_gaps(match):
        """All runs of frames inside the play ranges with no ATD ball: list of (first, last) inclusive."""
        out = []
        for a, b in Settings.PLAY_RANGES:
            miss = ~np.isfinite(match.bx[a:b])
            edges = np.diff(np.concatenate([[0], miss.astype(int), [0]]))
            for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                out.append((int(a + s), int(a + e - 1)))
        return out

    # ---- one gap ---------------------------------------------------------------------------------------------------
    def _seed(self, f, direction):
        """Seed at detected frame f; velocity (px per step in the tracking direction) from earlier detections."""
        x, y, z = self.det[f]
        for k in range(1, 9):
            g = f - direction * k          # frames *before* f in tracking order
            if g in self.det:
                return dict(x=x, y=y, vx=(x - self.det[g][0]) / k, vy=(y - self.det[g][1]) / k, z=z, slot=None)
        return dict(x=x, y=y, vx=0.0, vy=0.0, z=z, slot=None)

    def fill(self, s, e):
        """Track from the last detection before the gap forwards and from the first detection after it
        backwards. Returns an array (e-s+1, 11) with OUT_COLUMNS (NaN where a track gave no answer)."""
        n = e - s + 1
        res = np.full((n, len(self.OUT_COLUMNS)), np.nan)
        res[:, 0] = np.arange(s, e + 1)
        res[:, 1] = 0
        res[:, 6] = 0
        rng_start, rng_end = Settings.range_of(s) or (s, e + 1)
        a = next((f for f in range(s - 1, max(rng_start, s - self.SEED_SEARCH) - 1, -1) if f in self.det), None)
        b = next((f for f in range(e + 1, min(rng_end, e + self.SEED_SEARCH + 1)) if f in self.det), None)
        if a is not None:
            self._forward(res, a, s, e)
        if b is not None:
            self._backward(res, b, s, e)
        return res

    def _forward(self, res, a, s, e):
        tr = BallTracker(self.m, self.clf, self._seed(a, +1), self.params)
        f = a + 1
        while f <= e and not tr.lost:
            cs_list = self.reader.read_candidates(f, min(self.CHUNK, e - f + 1))
            for k, cs in enumerate(cs_list):
                o = tr.step(f + k, cs)
                if o is not None and s <= f + k <= e:
                    r = res[f + k - s]
                    r[1] = 1
                    r[2], r[3], r[4], r[5] = o["gx"], o["gy"], o["x"], o["y"]
                if tr.lost:
                    break
            f += len(cs_list)
            if not cs_list:
                break

    def _backward(self, res, b, s, e):
        tr = BallTracker(self.m, self.clf, self._seed(b, -1), self.params)
        hi = b - 1
        while hi >= s and not tr.lost:
            lo = max(s, hi - self.CHUNK + 1)
            cs_list = self.reader.read_candidates(lo, hi - lo + 1)
            for k in range(len(cs_list) - 1, -1, -1):
                f = lo + k
                o = tr.step(f, cs_list[k])
                if o is not None and s <= f <= e:
                    r = res[f - s]
                    r[6] = 1
                    r[7], r[8], r[9], r[10] = o["gx"], o["gy"], o["x"], o["y"]
                if tr.lost:
                    break
            hi = lo - 1

    # ---- every gap, in parallel (one file per gap in outputs/gaps/, so a run can be resumed) ------------------------
    @staticmethod
    def _make_context():
        match = Match()
        table = pd.read_csv(Settings.out("ball_height.csv"))
        table = table[table.detected == 1]
        detections = {int(f): (x, y, z) for f, x, y, z in zip(table.frame, table.x, table.y, table.z)}
        return GapFiller(match, BallClassifier.load(Settings.out("classifier.joblib")), detections)

    @staticmethod
    def _work(filler, gap):
        s, e = gap
        path = Settings.out("gaps", "g%06d.npy" % s)
        if os.path.exists(path):
            return gap
        np.save(path + ".tmp.npy", filler.fill(s, e))
        os.replace(path + ".tmp.npy", path)
        return gap

    @classmethod
    def run_gaps(cls, workers=6, limit=None, log=print):
        os.makedirs(Settings.out("gaps"), exist_ok=True)
        gaps = sorted(cls.find_gaps(Match()), key=lambda g: -(g[1] - g[0]))   # longest first keeps the pool balanced
        if limit:
            gaps = gaps[:limit]
        log("%d gaps, %d frames" % (len(gaps), sum(e - s + 1 for s, e in gaps)))
        ParallelJobs(cls._make_context, cls._work, workers, log).run(gaps, report_every=10)

    # ---- results ---------------------------------------------------------------------------------------------------
    @classmethod
    def summarize(cls, log=print):
        """outputs/gaps/*.npy -> outputs/gap_frames.csv (one row per gap frame, status agree/one/disagree/none)."""
        parts = []
        for f in sorted(glob.glob(Settings.out("gaps", "g*.npy"))):
            d = pd.DataFrame(np.load(f), columns=cls.OUT_COLUMNS)
            d["gap"] = int(os.path.basename(f)[1:7])
            d["gap_len"] = len(d)
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        dist = np.hypot(d.f_gx - d.b_gx, d.f_gy - d.b_gy)
        d["status"] = np.where((d.f_ok == 1) & (d.b_ok == 1), np.where(dist < cls.AGREE, "agree", "disagree"),
                               np.where((d.f_ok == 1) | (d.b_ok == 1), "one", "none"))
        d.to_csv(Settings.out("gap_frames.csv"), index=False)
        log("gaps: %d | frames: %d" % (d.gap.nunique(), len(d)))
        log("all gap frames: %s" % (d.status.value_counts(normalize=True) * 100).round(1).to_dict())
        bins = pd.cut(d.gap_len, [0, 125, 500, 1500, 10 ** 6], labels=["<5 s", "5-20 s", "20-60 s", ">60 s"])
        t = d.groupby(bins, observed=True).agg(gaps=("gap", "nunique"), frames=("frame", "size"))
        for s in ("agree", "one", "disagree", "none"):
            t[s + " %"] = (d.assign(k=d.status == s).groupby(bins, observed=True).k.mean() * 100).round(0)
        log(t.to_string())
        return d

    @classmethod
    def build_ball_track(cls, log=print):
        """-> outputs/ball_ground.csv with columns frame, gx, gy, source."""
        m = Match()
        frames = Settings.graded_frames()
        gx = np.full(m.N, np.nan)
        gy = np.full(m.N, np.nan)
        src = np.array([""] * m.N, dtype=object)
        atd = m.has_ball
        gx[atd], gy[atd], src[atd] = m.bx[atd], m.by[atd], "atd"
        cls._interpolate_short_gaps(m, gx, gy, src)
        cls._tracker_answers(gx, gy, src)
        out = pd.DataFrame(dict(frame=frames, gx=gx[frames], gy=gy[frames], source=src[frames]))
        out.to_csv(Settings.out("ball_ground.csv"), index=False)
        log("graded frames: %d" % len(out))
        log(out.source.value_counts().to_string())
        log("frames with a ball position: %d (%.1f%%)" % (out.gx.notna().sum(), 100 * out.gx.notna().mean()))
        return out

    @classmethod
    def _interpolate_short_gaps(cls, m, gx, gy, src):
        for s, e in cls.find_gaps(m):
            if e - s + 1 > cls.SHORT_GAP:
                continue
            a0, b1 = Settings.range_of(s)
            p0 = np.array([m.bx[s - 1], m.by[s - 1]]) if s - 1 >= a0 and np.isfinite(m.bx[s - 1]) else None
            p1 = np.array([m.bx[e + 1], m.by[e + 1]]) if e + 1 < b1 and np.isfinite(m.bx[e + 1]) else None
            for f in range(s, e + 1):
                if p0 is not None and p1 is not None:
                    q = p0 + (f - s + 1) / (e - s + 2) * (p1 - p0)
                else:
                    q = p0 if p0 is not None else p1
                if q is not None:
                    gx[f], gy[f] = q
                    src[f] = "interp"

    @classmethod
    def _tracker_answers(cls, gx, gy, src):
        g = pd.read_csv(Settings.out("gap_frames.csv"))
        g = g[g.gap_len > cls.SHORT_GAP]
        for r in g.itertuples():
            fr = int(r.frame)
            if r.status == "agree":
                gx[fr], gy[fr] = (r.f_gx + r.b_gx) / 2, (r.f_gy + r.b_gy) / 2
            elif r.status == "one":
                gx[fr], gy[fr] = (r.f_gx, r.f_gy) if r.f_ok == 1 else (r.b_gx, r.b_gy)
            src[fr] = r.status if r.status in ("agree", "one") else "none"
