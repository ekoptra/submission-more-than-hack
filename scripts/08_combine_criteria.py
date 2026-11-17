"""Stage 6 (AHP): combine the criteria with the AHP weights into P(H), P(A), P(D) per frame -> outputs/probabilities.csv.gz.
Usage: python scripts/08_combine_criteria.py"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.ahp import AHP  # noqa: E402
from helpers.settings import Settings  # noqa: E402


class CombineCriteriaScript:
    def run(self):
        probabilities = AHP.probabilities(pd.read_csv(Settings.out("criteria.csv.gz")))
        probabilities.to_csv(Settings.out("probabilities.csv.gz"), index=False)      # full precision: the HMM reads it back
        print("wrote", Settings.out("probabilities.csv.gz"), probabilities.shape)


if __name__ == "__main__":
    CombineCriteriaScript().run()
