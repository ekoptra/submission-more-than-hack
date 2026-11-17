"""Stage 7 (HMM): smooth the per-frame probabilities into stable possessions -> the submission file outputs/submission.csv.
Usage: python scripts/09_smooth_labels.py"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.possession_hmm import PossessionHMM  # noqa: E402
from helpers.settings import Settings  # noqa: E402


class SmoothLabelsScript:
    def run(self):
        labels = PossessionHMM().run(pd.read_csv(Settings.out("probabilities.csv.gz")))
        print("wrote", Settings.out("submission.csv"))
        print("share:", pd.Series(labels).value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    SmoothLabelsScript().run()
