"""Run the whole pipeline from the raw match data to outputs/submission.csv, stage by stage.
Usage: MTAH_DATA_DIR=<data folder> python scripts/run_all.py [--workers 6] [--from-stage 1]
Stages 2 and 4 read the video and take the longest; both are resumable, so an interrupted run can be started again.
--from-stage N starts at script number N (for example 5 to rebuild only the criteria and the labels)."""
import argparse
import glob
import os
import subprocess
import sys
import time


class RunAllScript:
    PARALLEL = ("02_detect_height.py", "04_fill_gaps.py")      # the stages that take --workers

    def __init__(self, workers, first_stage):
        self.workers, self.first_stage = workers, first_stage
        self.folder = os.path.dirname(os.path.abspath(__file__))

    def stages(self):
        scripts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(self.folder, "[0-9][0-9]_*.py")))
        return [s for s in scripts if int(s[:2]) >= self.first_stage]

    def run(self):
        t0 = time.time()
        for script in self.stages():
            args = [sys.executable, os.path.join(self.folder, script)]
            if script in self.PARALLEL:
                args += ["--workers", str(self.workers)]
            print("\n=== %s ===" % script, flush=True)
            started = time.time()
            if subprocess.run(args).returncode != 0:
                sys.exit("Stopped: %s failed (see the message above). Fix the cause and run again; finished jobs are kept." % script)
            print("--- %s done in %.0f s" % (script, time.time() - started), flush=True)
        print("\nAll stages finished in %.0f s. Submission: outputs/submission.csv" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--from-stage", type=int, default=1)
    a = ap.parse_args()
    RunAllScript(a.workers, a.from_stage).run()
