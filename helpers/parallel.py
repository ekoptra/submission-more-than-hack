"""Running independent jobs in a pool of worker processes.

Each worker builds its own context once (match data, video handle, classifier ...), because video handles cannot be
shared between processes. `make_context` and `work` must be importable (static methods of a class at module level).
"""
import time
from functools import partial
from multiprocessing import Pool

import cv2


class ParallelJobs:
    _context = None

    def __init__(self, make_context, work, workers=6, log=print):
        self.make_context, self.work, self.workers, self.log = make_context, work, workers, log

    @staticmethod
    def _init(make_context):
        cv2.setNumThreads(1)
        ParallelJobs._context = make_context()

    @staticmethod
    def _call(work, job):
        return work(ParallelJobs._context, job)

    def run(self, jobs, report_every=20):
        """Run `work(context, job)` for every job; results come back as they finish."""
        results, t0 = [], time.time()
        with Pool(self.workers, initializer=ParallelJobs._init, initargs=(self.make_context,)) as pool:
            for done, res in enumerate(pool.imap_unordered(partial(ParallelJobs._call, self.work), jobs), 1):
                results.append(res)
                if done % report_every == 0 or done == len(jobs):
                    el = time.time() - t0
                    self.log("%d/%d jobs | %.0f s elapsed | est. %.0f s remaining"
                             % (done, len(jobs), el, el / done * (len(jobs) - done)))
        self.log("finished in %.0f s" % (time.time() - t0))
        return results
