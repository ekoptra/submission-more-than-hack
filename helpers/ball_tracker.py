"""Following the ball through a stretch with no ATD ball position, starting from a known image position.

State: the ball's pixel position and its pixel velocity per frame, plus the last known height (metres).
Frame by frame:
  * predict the next pixel position with the current velocity;
  * accept only candidates inside a gate around the prediction; the gate grows with every consecutive
    frame without an accepted candidate (R0 + GROW * misses, capped), because the ball keeps moving;
  * among the gated candidates take the best by  log(classifier score) - 0.5 * (distance / sigma)^2;
  * a candidate on a player's body is accepted if the previous pick was on that same player (a ball
    being carried or held) or if its classifier score is high (P_BODY); otherwise it is most likely
    a boot or a shirt;
  * after MAX_MISS consecutive misses the track is declared lost.
The ground position of a pick needs a height: we keep the last height we knew (a ball that was on the
ground stays on the ground; one that was in the air is assumed to keep that height).
"""
import numpy as np

from .camera import Camera


class BallTracker:
    DEFAULTS = dict(R0=30.0, GROW=12.0, R_MAX=160.0,   # gate radius (px): base, growth per missed frame, cap
                    MAX_MISS=25,                       # consecutive misses before giving up (1 s at 25 fps)
                    P_MIN=0.15,                        # classifier score below which a candidate is ignored
                    V_DAMP=0.9,                        # velocity damping per missed frame
                    P_BODY=0.5)                        # score needed for a candidate on another player's body

    def __init__(self, match, classifier, seed, params=None):
        """seed: dict(x, y, vx, vy, z, slot) at the last known position; params override DEFAULTS."""
        self.m, self.clf = match, classifier
        self.P = dict(self.DEFAULTS)
        if params:
            self.P.update(params)
        self.x, self.y, self.vx, self.vy = seed["x"], seed["y"], seed["vx"], seed["vy"]
        self.z, self.slot = seed["z"], seed.get("slot")
        self.misses, self.lost = 0, False

    def step(self, f, cs):
        """Returns dict(x, y, z, gx, gy, p, dist) when the ball is found, else None."""
        P = self.P
        if self.lost:
            return None
        px, py = self.x + self.vx, self.y + self.vy
        r = min(P["R_MAX"], P["R0"] + P["GROW"] * self.misses)
        pr = self.clf.proba(cs)
        best, bs = None, -1e9
        for c, p in zip(cs, pr):
            if p < P["P_MIN"]:
                continue
            d = float(np.hypot(c["x"] - px, c["y"] - py))
            if d > r:
                continue
            if c["slot"] is not None and c["slot"] != self.slot and p < P["P_BODY"]:
                continue          # on another player's body/boot: needs a high score to be believed
            sc = np.log(p + 1e-3) - 0.5 * (d / (r / 2)) ** 2
            if sc > bs:
                best, bs = (c, p, d), sc
        if best is None:
            self.misses += 1
            self.vx, self.vy = self.vx * P["V_DAMP"], self.vy * P["V_DAMP"]
            self.x, self.y = px, py
            if self.misses > P["MAX_MISS"]:
                self.lost = True
            return None
        c, p, d = best
        n = self.misses + 1
        self.vx = 0.5 * self.vx + 0.5 * (c["x"] - self.x) / n
        self.vy = 0.5 * self.vy + 0.5 * (c["y"] - self.y) / n
        self.x, self.y, self.slot, self.misses = c["x"], c["y"], c["slot"], 0
        gx, gy = Camera(self.m.H[f]).ground_from_pixel(self.x, self.y, self.z)
        return dict(x=self.x, y=self.y, z=self.z, gx=gx, gy=gy, p=float(p), dist=d)
