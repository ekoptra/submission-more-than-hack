"""The provided match data: ATD tracking, per-frame homography, track-slot teams and Smart Tagging."""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .settings import Settings


class Match:
    """Everything the pipeline reads about the game.

    X, Y   (n, 46)  player positions in metres (NaN where the slot has no position)
    B      (n, 3)   raw ball (x, y, speed), normalised; NaN where the ball is missing
    bx, by (n,)     ball ground position in metres (ATD)
    H      (n,3,3)  homography, normalised pitch -> image pixels
    has_ball (n,)   ATD ball present and frame inside a play range
    phases          Smart Tagging phases (Metrica's automatic annotation: an input, not ground truth)
    Frame index = video frame of the tactical video (row i of the ATD text file is video frame i).
    """

    HOME_TAG, AWAY_TAG = "barcelona", "atletic club"      # team spelling in the Smart Tagging file

    def __init__(self, cache_dir=None):
        paths = Settings.asset_paths()
        cache_dir = cache_dir or Settings.OUT
        cache = os.path.join(cache_dir, "atd_cache.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            Xn, Yn, B = z["X"], z["Y"], z["B"]
        else:
            Xn, Yn, B = self._parse_atd(paths["atd"])
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache, X=Xn, Y=Yn, B=B)
        L, W = Settings.PITCH_L, Settings.PITCH_W
        self.X, self.Y, self.B = Xn * L, Yn * W, B
        self.N = len(B)
        self.bx, self.by = B[:, 0] * L, B[:, 1] * W
        table = pd.read_csv(paths["homography"])
        cols = ["h%d%d" % (r, c) for r in range(3) for c in range(3)]
        self.H = table[cols].values.astype(np.float64).reshape(-1, 3, 3)
        self.video_path = paths["video"]
        in_play = np.zeros(self.N, bool)
        for a, b in Settings.PLAY_RANGES:
            in_play[a:b] = True
        self.in_play = in_play
        self.has_ball = np.isfinite(self.bx) & in_play
        self.phases = self._read_smart_tagging(paths["smart_tagging"])

    @staticmethod
    def _parse_atd(path):
        """ATD text: `frame_id:x,y,s;x,y,s;...(46 slots):bx,by,bs` -> X, Y (46 slots, normalised), B (n, 3)."""
        X, Y, B = [], [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split(":")
                a = np.array([[float(v) for v in s.split(",")] for s in parts[1].split(";")])
                X.append(a[:, 0])
                Y.append(a[:, 1])
                B.append([float(v) for v in parts[2].split(",")])
        return np.array(X, np.float32), np.array(Y, np.float32), np.array(B, np.float32)

    @staticmethod
    def _read_smart_tagging(path):
        phases = []
        for ins in ET.parse(path).getroot().iter("instance"):
            tags = {lab.findtext("group"): lab.findtext("text") for lab in ins.findall("label")}
            phases.append(dict(code=ins.findtext("code").strip().lower(),
                               a=int(round(float(ins.findtext("start")) * Settings.FPS)),
                               b=int(round(float(ins.findtext("end")) * Settings.FPS)), tags=tags))
        return phases

    @staticmethod
    def home_mask():
        """Boolean array over the 46 slots: True for the home team (the first team of the ATD metadata, the FIFA/EPTS
        convention). A slot is a tracklet, not a person: it can swap identity during the match."""
        root = ET.parse(Settings.asset_paths()["atd_meta"]).getroot()
        teams = [t.get("id") for t in root.iter("Team")]
        return np.array([p.get("teamId") == teams[0] for p in root.iter("Player")])

    def set_piece_mask(self):
        """True on frames inside a Smart Tagging SET PIECES phase."""
        mask = np.zeros(self.N, bool)
        for p in self.phases:
            if p["code"] == "set pieces":
                mask[p["a"]:p["b"]] = True
        return mask

    def team_prior(self):
        """Share of home among the running phases with a real team tag (0.5 where there is none)."""
        total, count = np.zeros(self.N), np.zeros(self.N)
        for p in self.phases:
            team = (p["tags"].get("Team") or "").strip().lower()
            if team in (self.HOME_TAG, self.AWAY_TAG):
                total[p["a"]:p["b"]] += 1.0 if team == self.HOME_TAG else 0.0
                count[p["a"]:p["b"]] += 1
        return np.where(count > 0, total / np.maximum(count, 1), 0.5)

    def kickoff_window(self, half):
        """(start, end) frames of the 'Kick Off (Start)' set piece of '1st Half' / '2nd Half'."""
        ko = [p for p in self.phases if p["code"] == "set pieces"
              and (p["tags"].get("Type") or "").lower() == "kick off (start)" and p["tags"].get("Half") == half][0]
        return ko["a"], ko["b"]
