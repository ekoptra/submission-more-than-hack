"""Stage 2b: merge outputs/blocks/*.npy into outputs/ball_height.csv and print per-block statistics.
Usage: MTAH_DATA_DIR=<data folder> python scripts/03_aggregate_height.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.height_detector import HeightDetector  # noqa: E402


class AggregateHeightScript:
    def run(self):
        HeightDetector.aggregate()


if __name__ == "__main__":
    AggregateHeightScript().run()
