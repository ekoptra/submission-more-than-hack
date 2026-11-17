"""Stage 3: run the two-way tracker over every real gap in the ATD ball (needs ball_height.csv and the classifier).
One file per gap in outputs/gaps/ (existing files are skipped, so an interrupted run can be resumed).
Usage: MTAH_DATA_DIR=<data folder> python scripts/04_fill_gaps.py [--workers 6] [--limit N]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.gap_filler import GapFiller  # noqa: E402


class FillGapsScript:
    def run(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--workers", type=int, default=6)
        ap.add_argument("--limit", type=int, default=None)
        a = ap.parse_args()
        GapFiller.run_gaps(a.workers, a.limit, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    FillGapsScript().run()
