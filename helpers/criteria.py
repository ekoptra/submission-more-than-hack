"""Stage 5: the per-frame criteria matrix (docs/technical_document.md, section 5).

Inputs, all produced by an algorithm on the provided game (no labels): ATD tracking, outputs/ball_ground.csv,
outputs/ball_height.csv and the Smart Tagging XML. No ground truth (public or private) is read.
Output: outputs/criteria.csv.gz, one row per graded frame; NaN = criterion not computable on that frame.

Dead-ball criteria: D1 collective speed, D2 ball beyond the lines, D3 players still, D5 ball untracked, D6 Smart Tagging
set piece, D7 lone still player at a touchline or corner. (D4, "ball almost still", is not used: the ball moves during a
stoppage.) Possession criteria: T1 closeness, T2 local dominance, T3 last controller, T4 Smart Tagging team, T5 team
shape, T6 motion match, with the control factor c that lowers the weight of closeness when the ball is in flight.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .match import Match
from .scoring import Scoring
from .settings import Settings
from .signals import Signals


class Criteria:
    COLUMNS = ["D1", "D2", "D3", "D5", "D6", "D7", "T1", "T2", "T3", "T4", "T5", "T6", "c", "e", "v_ball", "v_rel", "z", "d_min"]
    CHUNK = 20000
    CLEAR_CONTROL = 0.6        # c above this: a player clearly controls the ball
    MEMORY_S = 3.0             # how long the last clear controller still counts (T3)
    SHAPE_BLOCK = 7500         # T5: frames per cross-fitting block (5 minutes)
    SHAPE_TRAIN_FRACTION = 0.2 # T5: share of the training frames used (consecutive frames are nearly identical)
    SHAPE_SEED = 0

    def __init__(self, log=print):
        self.log = log
        self.m = Match()
        self.F = Settings.graded_frames()
        self.n, self.N = len(self.F), self.m.N
        self.signals = Signals(self.m, self.F, Match.home_mask())
        self.players, self.ball, self.rel = self.signals.players, self.signals.ball, self.signals.relations
        self.c = self.control_factor()
        self.e = Scoring.ramp(self.ball.speed, 0.5, 2.0)      # the ball moves enough for motion evidence (T6)

    def build(self):
        """Compute every criterion and write outputs/criteria.csv.gz."""
        values = dict(D1=self.collective_speed(), D2=self.ball_outside_line(), D3=self.players_still(),
                      D5=self.ball_untracked(), D6=self.set_piece_tag(), D7=self.restart_taker(),
                      T1=self.closeness(), T2=self.local_dominance(), T3=self.last_controller(), T4=self.tag_prior(),
                      T5=self.team_shape(), T6=self.motion_match(), c=self.c, e=self.e,
                      v_ball=self.ball.speed, v_rel=self.rel.v_rel, z=self.ball.z, d_min=self.rel.d_min)
        table = pd.DataFrame({"frame": self.F, "source": self.ball.source})
        for name in self.COLUMNS:
            table[name] = values[name]
        path = Settings.out("criteria.csv.gz")
        table.to_csv(path, index=False, float_format="%.5g")
        self.log("wrote %s %s" % (path, table.shape))
        return table

    # ---- dead-ball criteria ----------------------------------------------------------------------------------------
    def collective_speed(self):
        """D1: median player speed, mean over +-2 s, ramp 1.33 -> 0.53 m/s."""
        series = np.full(self.N, np.nan)
        series[self.F] = self.players.median_speed
        return Scoring.ramp(Scoring.window_mean(series, 50)[self.F], 1.33, 0.53)

    def ball_outside_line(self):
        """D2: distance of the ball beyond the pitch lines (mean over +-1 s, ramp 0.5 -> 2.0 m) times a slow-ball factor."""
        L, W = Settings.PITCH_L, Settings.PITCH_W
        bx, by, v = self.ball.bx_all, self.ball.by_all, self.ball.speed
        outside = np.hypot(np.maximum(0, np.maximum(-bx, bx - L)), np.maximum(0, np.maximum(-by, by - W)))
        outside = Scoring.window_mean(outside, 25)[self.F]
        score = Scoring.ramp(outside, 0.5, 2.0) * np.where(np.isfinite(v), Scoring.ramp(v, 3, 0.5), 1.0)
        return np.where(np.isfinite(outside), score, np.nan)

    def players_still(self):
        """D3: share of players slower than 0.5 m/s, ramp 0.14 -> 0.50."""
        return Scoring.ramp(self.players.share_still, 0.14, 0.50)

    def ball_untracked(self):
        """D5: share of frames in +-1 s without any ball position."""
        return Scoring.window_mean(self.ball.unknown_all, 25)[self.F]

    def set_piece_tag(self):
        """D6: 1 inside a Smart Tagging SET PIECES phase."""
        return self.m.set_piece_mask()[self.F].astype(float)

    def restart_taker(self):
        """D7: a still, isolated player at a touchline or corner flag (mean over +-1 s of the best player score)."""
        L, W = Settings.PITCH_L, Settings.PITCH_W
        flags = [(0, 0), (0, W), (L, 0), (L, W)]
        p, best = self.players, np.full(self.n, np.nan)
        for s in range(0, self.n, self.CHUNK):
            sl = slice(s, min(self.n, s + self.CHUNK))
            X, Y = p.X[sl], p.Y[sl]
            dd = np.hypot(X[:, :, None] - X[:, None, :], Y[:, :, None] - Y[:, None, :])
            dd[:, np.arange(46), np.arange(46)] = np.inf
            gap = np.nanmin(np.where(np.isfinite(dd), dd, np.inf), axis=2)
            gap = np.where(np.isfinite(X), gap, np.nan)
            line = np.minimum(np.abs(Y), np.abs(Y - W))
            for fx, fy in flags:
                line = np.minimum(line, np.hypot(X - fx, Y - fy))
            score = Scoring.ramp(p.speed[sl], 0.5, 0.1) * Scoring.ramp(gap, 2, 4) * Scoring.ramp(line, 2.0, 0.5)
            best[sl] = np.nanmax(np.where(np.isfinite(score), score, -1), axis=1)
        best = np.where(best < 0, np.nan, best)
        series = np.full(self.N, np.nan)
        series[self.F] = best
        return Scoring.window_mean(series, 25)[self.F]

    # ---- possession criteria ---------------------------------------------------------------------------------------
    def control_factor(self):
        """c = c_distance * c_relative_speed * c_height; unknown speed or height count as 0.7."""
        r = Scoring.ramp
        c_speed = np.where(np.isfinite(self.rel.v_rel), r(self.rel.v_rel, 8, 2), 0.7)
        c_height = np.where(np.isfinite(self.ball.z), r(self.ball.z, 3.0, 1.5), 0.7)
        return r(self.rel.d_min, 3.0, 1.0) * c_speed * c_height

    def closeness(self):
        """T1: relative closeness of the nearest home and away players, weighted by c."""
        rel = self.rel
        with np.errstate(invalid="ignore", divide="ignore"):
            s1 = np.where(rel.d_home + rel.d_away > 0, rel.d_away / (rel.d_home + rel.d_away), 0.5)
        return np.where(rel.known, 0.5 + self.c * (s1 - 0.5), np.nan)

    def local_dominance(self):
        """T2: players of each team within 3 m of the ball, weighted by c."""
        rel = self.rel
        share = (rel.n_home + 0.5) / (rel.n_home + rel.n_away + 1)
        return np.where(rel.known, 0.5 + self.c * (share - 0.5), np.nan)

    def last_controller(self):
        """T3: the team that last clearly controlled the ball, fading linearly over 3 s."""
        n, rel = self.n, self.rel
        clear = rel.known & (self.c > self.CLEAR_CONTROL)
        sigma = np.where(rel.nearest_is_home, 1.0, -1.0)
        last = np.maximum.accumulate(np.where(clear, np.arange(n), -10 ** 9))     # index into the graded frames
        second_half = np.searchsorted(self.F, Settings.PLAY_RANGES[1][0])           # restart the carry at the second half
        last[second_half:] = np.where(last[second_half:] >= second_half, last[second_half:], -10 ** 9)
        dt = (np.arange(n) - last) / Settings.FPS
        valid = (last >= 0) & (dt <= self.MEMORY_S)
        return np.where(valid, 0.5 + 0.5 * sigma[np.clip(last, 0, n - 1)] * np.maximum(0, 1 - dt / self.MEMORY_S), np.nan)

    def tag_prior(self):
        """T4: Smart Tagging team of the running phases (0.5 when there is none)."""
        return self.m.team_prior()[self.F]

    def motion_match(self):
        """T6: which team has the player moving together with the ball (needs a ball velocity)."""
        match = self.signals.motion_match()
        best = {}
        for name, mask in (("home", self.players.home), ("away", ~self.players.home)):
            sel = np.where(mask, match, np.nan)
            best[name] = np.nanmax(sel, axis=1, initial=0.0, where=np.isfinite(sel))
        return np.where(self.rel.known & np.isfinite(self.ball.speed),
                        0.5 + 0.5 * self.e * (best["home"] - best["away"]), np.nan)

    # ---- T5: team shape ---------------------------------------------------------------------------------------------
    def attack_directions(self):
        """+1 if the home team attacks towards +x, for each half; decided by the side each team occupies at kick-off."""
        out = []
        for half in ("1st Half", "2nd Half"):
            a, b = self.m.kickoff_window(half)
            home_mean = np.nanmean(self.players.X_all[a:b][:, self.players.home])
            away_mean = np.nanmean(self.players.X_all[a:b][:, ~self.players.home])
            out.append(1 if home_mean < away_mean else -1)
        return out

    def shape_features(self, directions):
        """(n, 9): centroid, depth, width of each team and their differences; the rearmost player (goalkeeper) is left out."""
        L = Settings.PITCH_L
        half_of = np.where(self.F < Settings.PLAY_RANGES[0][1], 0, 1)
        feats = np.full((self.n, 9), np.nan)
        for hf in (0, 1):
            idx = np.where(half_of == hf)[0]
            xa = self.players.X[idx] if directions[hf] == 1 else L - self.players.X[idx]
            ya = self.players.Y[idx]
            cols = []
            for mask in (self.players.home, ~self.players.home):
                xt, yt = np.where(mask, xa, np.nan), np.where(mask, ya, np.nan)
                rear = np.nanargmin(np.where(np.isfinite(xt), xt, np.inf), axis=1)
                xt[np.arange(len(idx)), rear] = np.nan
                yt[np.arange(len(idx)), rear] = np.nan
                cols.append((np.nanmean(xt, axis=1),
                             np.nanpercentile(xt, 90, axis=1) - np.nanpercentile(xt, 10, axis=1),
                             np.nanpercentile(yt, 90, axis=1) - np.nanpercentile(yt, 10, axis=1)))
            (c1, d1, w1), (c2, d2, w2) = cols
            feats[idx] = np.column_stack([c1, d1, w1, c2, d2, w2, c1 - c2, d1 - d2, w1 - w2])
        return feats

    def team_shape(self):
        """T5: probability that the home team has the ball, from the shape of the two teams only (no ball needed).

        A logistic regression on team centroid, depth and width in the attacking direction. The training label is made
        automatically: the team of the player nearest to the ATD ball on frames where the ATD has the ball. The match is
        cut into 5-minute blocks and each block is predicted by a model trained on the other blocks, so no frame is
        predicted by a model that has seen its own label. No hand labels."""
        feats = self.shape_features(self.attack_directions())
        label = np.where(self.rel.nearest_is_home, 1, 0)
        usable = (self.ball.source == "atd") & self.rel.known
        half_of = np.where(self.F < Settings.PLAY_RANGES[0][1], 0, 1)
        starts = np.where(half_of == 0, Settings.PLAY_RANGES[0][0], Settings.PLAY_RANGES[1][0])
        block = (self.F - starts) // self.SHAPE_BLOCK + 2 * half_of * 100
        ok = np.isfinite(feats).all(axis=1)
        out = np.full(self.n, np.nan)
        agreement = []
        rng = np.random.default_rng(self.SHAPE_SEED)
        for b in np.unique(block):
            train = np.where(usable & ok & (block != b))[0]
            train = train[rng.random(len(train)) < self.SHAPE_TRAIN_FRACTION]
            scaler = StandardScaler().fit(feats[train])
            model = LogisticRegression(C=1.0, max_iter=500).fit(scaler.transform(feats[train]), label[train])
            test = np.where((block == b) & ok)[0]
            out[test] = model.predict_proba(scaler.transform(feats[test]))[:, 1]
            check = test[usable[test]]
            agreement.append(((out[check] > 0.5) == (label[check] == 1)).mean() if len(check) else np.nan)
        self.log("T5 cross-fitted agreement with the automatic label (nearest player's team): %.3f over %d blocks (mean of blocks)"
                 % (np.nanmean(agreement), len(agreement)))
        return out
