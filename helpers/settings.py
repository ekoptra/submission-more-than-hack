"""Paths and constants shared by the whole pipeline."""
import os

import numpy as np


class Settings:
    """Match constants and locations. The match assets are the competition's gated download: point the
    environment variable MTAH_DATA_DIR at the folder that contains `tactical/`."""

    GAME = "BAR-ATH"
    PITCH_L, PITCH_W = 105.3, 68.0          # metres
    FPS = 25
    N_FRAMES = 199319
    PLAY_RANGES = [(11185, 90652), (110808, 190198)]   # graded frames: first and second half, [start, end)
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT = os.path.join(ROOT, "outputs")

    @classmethod
    def load_env_file(cls):
        """Read `KEY=VALUE` lines from `.env` next to the scripts' parent folder (if it exists) into the environment.
        Variables that are already set in the environment win. Lines starting with # are comments."""
        path = os.path.join(cls.ROOT, ".env")
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

    @classmethod
    def data_dir(cls):
        cls.load_env_file()
        folder = os.environ.get("MTAH_DATA_DIR")
        if not folder:
            raise RuntimeError("Set MTAH_DATA_DIR (environment variable or a .env file, see .env.example) to the folder that contains tactical/ "
                               "(e.g. the challenge repository's data/ folder).")
        return folder

    @classmethod
    def asset_paths(cls):
        tactical = os.path.join(cls.data_dir(), "tactical")
        name = cls.GAME + "_tactical"
        return dict(video=os.path.join(tactical, name + ".mp4"),
                    atd=os.path.join(tactical, name + "_atd.txt"),
                    atd_meta=os.path.join(tactical, name + "_atd.xml"),
                    homography=os.path.join(tactical, name + "_homography.csv"),
                    smart_tagging=os.path.join(tactical, name + "_smart_tagging.xml"))

    @classmethod
    def out(cls, *parts):
        """Path inside the (git-ignored) outputs folder; the folder is created when missing."""
        os.makedirs(cls.OUT, exist_ok=True)
        return os.path.join(cls.OUT, *parts)

    @classmethod
    def graded_frames(cls):
        return np.concatenate([np.arange(a, b) for a, b in cls.PLAY_RANGES])

    @classmethod
    def range_of(cls, frame):
        """(start, end) of the play range that contains `frame`, or None."""
        return next(((a, b) for a, b in cls.PLAY_RANGES if a <= frame < b), None)
