"""Stage 7: Hidden Markov Model (HMM) smoothing of the per-frame probabilities, and the submission file
(docs/technical_document.md, section 7).

States H, A, D; the emission log-likelihood of a frame is the log of its probability (from the AHP combination). A state
is kept with probability 1 - 1/(fps * T) per frame, T being its expected stay in seconds, and the rest is split over the
other states: T = 3 s for H and A (the challenge definition: a player keeps the ball for a few seconds) and T = 10 s for D
(a stoppage lasts several seconds to tens of seconds). Entering D has the same probability as switching between H and A.
The most probable path comes from the Viterbi algorithm, separately for each half (the halftime break is not graded).
No parameter comes from labelled data.
"""
import numpy as np
import pandas as pd

from .settings import Settings


class PossessionHMM:
    STATES = np.array(["H", "A", "D"])
    EPS = 1e-3          # floor on a probability so one frame cannot veto a state for ever
    UNGRADED = "X"      # frames that are not graded: whatever is written there is ignored

    def __init__(self, stay_s=3.0, dead_stay_s=10.0):
        switch = 1.0 / (Settings.FPS * stay_s)
        leave_dead = 1.0 / (Settings.FPS * dead_stay_s)
        move = np.log(switch / 2)
        self.log_trans = np.array([
            [np.log(1 - switch - switch), move, np.log(switch)],            # from H to H, A, D
            [move, np.log(1 - switch - switch), np.log(switch)],            # from A
            [np.log(leave_dead / 2), np.log(leave_dead / 2), np.log(1 - leave_dead)]])   # from D

    def viterbi(self, log_emission):
        """(n, 3) emission log-likelihoods -> most probable state path (0 = H, 1 = A, 2 = D)."""
        n = len(log_emission)
        score = log_emission[0].copy()
        back = np.zeros((n, 3), np.int8)
        for t in range(1, n):
            cand = score[:, None] + self.log_trans
            back[t] = cand.argmax(axis=0)
            score = cand.max(axis=0) + log_emission[t]
        path = np.zeros(n, np.int8)
        path[-1] = score.argmax()
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def decode(self, frames, probabilities):
        """Labels ('H', 'A', 'D') for the graded frames, a path per half. probabilities: (n, 3) columns H, A, D."""
        log_emission = np.log(np.clip(probabilities, self.EPS, 1.0))
        path = np.zeros(len(frames), np.int8)
        for start, end in Settings.PLAY_RANGES:
            half = (frames >= start) & (frames < end)
            path[half] = self.viterbi(log_emission[half])
        return self.STATES[path]

    def write(self, frames, labels, path):
        """One `frame,label` row per video frame (199,319 rows): H / A / D on graded frames, X elsewhere."""
        full = np.full(Settings.N_FRAMES, self.UNGRADED, dtype="<U1")
        full[frames] = labels
        pd.DataFrame({"frame": np.arange(Settings.N_FRAMES), "label": full}).to_csv(path, index=False)
        return path

    def run(self, probabilities):
        """probabilities (DataFrame from outputs/probabilities.csv.gz) -> outputs/submission.csv. Returns the labels."""
        frames = probabilities.frame.values
        labels = self.decode(frames, probabilities[["P_H", "P_A", "P_D"]].values)
        self.write(frames, labels, Settings.out("submission.csv"))
        return labels
