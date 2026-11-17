"""Stage 1: train the ball-candidate classifier on 12 windows of the match and save outputs/classifier.joblib.
Usage: MTAH_DATA_DIR=<data folder> python scripts/01_train_classifier.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.ball_classifier import BallClassifier  # noqa: E402
from helpers.match import Match  # noqa: E402
from helpers.settings import Settings  # noqa: E402


class TrainClassifierScript:
    def run(self):
        classifier = BallClassifier.train(Match(), log=lambda s: print(s, flush=True))
        classifier.save(Settings.out("classifier.joblib"))
        print("saved outputs/classifier.joblib")


if __name__ == "__main__":
    TrainClassifierScript().run()
