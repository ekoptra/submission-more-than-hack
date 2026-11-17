"""Per-frame measurements on the graded frames: ball (position, source, height, velocity), players (positions, speeds,
velocity vectors) and the relations between them (distances, counts, relative speed)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .scoring import Scoring
from .settings import Settings


class Signals:
    PLAYER_SPEED_HALF_WINDOW = 12     # +-0.5 s for player speed
    VELOCITY_HALF_WINDOW = 5          # +-0.2 s for velocity vectors
    NEAR_RADIUS = 3.0                 # metres

    def __init__(self, match, frames, home):
        """Reads outputs/ball_ground.csv and outputs/ball_height.csv. Builds three namespaces over the graded frames:
        self.ball, self.players and self.relations."""
        self.ball = self._ball(match, frames)
        self.players = self._players(match, frames, home)
        self.relations = self._relations(self.ball, self.players)

    def _ball(self, match, F):
        N, fps, K = match.N, Settings.FPS, self.VELOCITY_HALF_WINDOW
        g = pd.read_csv(Settings.out("ball_ground.csv"))
        bx = np.full(N, np.nan)
        by = np.full(N, np.nan)
        src = np.full(N, "none", object)
        bx[g.frame.values], by[g.frame.values], src[g.frame.values] = g.gx.values, g.gy.values, g.source.values
        has = src != "none"
        bx[~has] = np.nan
        by[~has] = np.nan
        reliable = np.isin(src, ["atd", "interp"])      # only these give a trustworthy velocity
        rx, ry = np.where(reliable, bx, np.nan), np.where(reliable, by, np.nan)
        heights = pd.read_csv(Settings.out("ball_height.csv"))
        heights = heights[heights.detected == 1]
        z = np.full(N, np.nan)
        z[heights.frame.values] = np.maximum(heights.z.values, 0.0)
        ux = (rx[F + K] - rx[F - K]) / (2 * K / fps)
        uy = (ry[F + K] - ry[F - K]) / (2 * K / fps)
        return SimpleNamespace(bx_all=bx, by_all=by, source_all=src,                       # over all video frames
                               x=bx[F], y=by[F], source=src[F], z=z[F],                    # over graded frames
                               ux=ux, uy=uy, speed=np.hypot(ux, uy), unknown_all=(src == "none").astype(float))

    def _players(self, match, F, home):
        fps, Wv, K = Settings.FPS, self.PLAYER_SPEED_HALF_WINDOW, self.VELOCITY_HALF_WINDOW
        speed = np.hypot(match.X[F + Wv] - match.X[F - Wv], match.Y[F + Wv] - match.Y[F - Wv]) / (2 * Wv / fps)
        return SimpleNamespace(
            home=home, X_all=match.X, X=match.X[F], Y=match.Y[F], speed=speed,           # (n, 46)
            ux=(match.X[F + K] - match.X[F - K]) / (2 * K / fps),
            uy=(match.Y[F + K] - match.Y[F - K]) / (2 * K / fps),
            median_speed=np.nanmedian(np.where(np.isfinite(speed), speed, np.nan), axis=1),
            share_still=np.nanmean(np.where(np.isfinite(speed), speed < 0.5, np.nan), axis=1))

    @staticmethod
    def _nearest(dist, mask):
        sel = np.where(mask, dist, np.nan)
        out = np.nanmin(sel, axis=1, initial=np.inf, where=np.isfinite(sel))
        return np.where(np.isfinite(out), out, np.nan)

    def _relations(self, ball, players):
        home = players.home
        dist = np.hypot(players.X - ball.x[:, None], players.Y - ball.y[:, None])      # NaN where ball or slot missing
        dh, da = self._nearest(dist, home), self._nearest(dist, ~home)
        d_min = np.fmin(dh, da)
        # speed of the ball relative to the nearest player: a carried ball moves with its carrier, a passed ball does not
        near = np.where(np.isfinite(dist), dist, np.inf).argmin(axis=1)
        ar = np.arange(len(ball.x))
        v_rel = np.hypot(ball.ux - players.ux[ar, near], ball.uy - players.uy[ar, near])
        return SimpleNamespace(
            dist=dist, d_home=dh, d_away=da, d_min=d_min,
            n_home=np.sum(np.where(home, dist < self.NEAR_RADIUS, False), axis=1),
            n_away=np.sum(np.where(~home, dist < self.NEAR_RADIUS, False), axis=1),
            known=np.isfinite(ball.x) & np.isfinite(dh) & np.isfinite(da),
            v_rel=np.where(np.isfinite(d_min), v_rel, np.nan), nearest_is_home=dh <= da)

    def motion_match(self):
        """Per slot within 3 m of the ball: ramp of the velocity-vector difference to the ball (1 below 1 m/s, 0 above 4 m/s)."""
        p, b = self.players, self.ball
        du = np.hypot(p.ux - b.ux[:, None], p.uy - b.uy[:, None])
        return np.where((self.relations.dist < self.NEAR_RADIUS) & np.isfinite(du), Scoring.ramp(du, 4.0, 1.0), np.nan)
