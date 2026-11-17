"""Stage 3b: collect the gap results into outputs/gap_frames.csv and print how many gap frames get an answer.
Usage: MTAH_DATA_DIR=<data folder> python scripts/05_summarize_gaps.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.gap_filler import GapFiller  # noqa: E402


class SummarizeGapsScript:
    def run(self):
        GapFiller.summarize()


if __name__ == "__main__":
    SummarizeGapsScript().run()
