"""Stage 2: run the ball-height detector over the whole match, block by block, in parallel.
One file per block in outputs/blocks/ (existing blocks are skipped, so an interrupted run can be resumed).
Usage: MTAH_DATA_DIR=<data folder> python scripts/02_detect_height.py [--workers 6] [--limit N] [--start F]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.height_detector import HeightDetector  # noqa: E402


class DetectHeightScript:
    def run(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--workers", type=int, default=6)
        ap.add_argument("--limit", type=int, default=None, help="only the first N blocks (timing)")
        ap.add_argument("--start", type=int, default=None, help="only this block start frame")
        a = ap.parse_args()
        HeightDetector.run_blocks(a.workers, a.limit, a.start, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    DetectHeightScript().run()
