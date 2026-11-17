"""Stage 5: the per-frame criteria matrix (dead-ball and possession criteria) -> outputs/criteria.csv.gz.
Usage: MTAH_DATA_DIR=<data folder> python scripts/07_compute_criteria.py     (about 5 minutes)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.criteria import Criteria  # noqa: E402


class ComputeCriteriaScript:
    def run(self):
        Criteria().build()


if __name__ == "__main__":
    ComputeCriteriaScript().run()
