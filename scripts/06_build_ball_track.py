"""Stage 4: one ball ground position per graded frame, with the source of each value -> outputs/ball_ground.csv.
Usage: MTAH_DATA_DIR=<data folder> python scripts/06_build_ball_track.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.gap_filler import GapFiller  # noqa: E402


class BuildBallTrackScript:
    def run(self):
        GapFiller.build_ball_track()


if __name__ == "__main__":
    BuildBallTrackScript().run()
