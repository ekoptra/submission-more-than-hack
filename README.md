# Ball possession (H / A / D) from tactical-video tracking

More Than A Hack 2027, phase 1. One label per frame of the tactical video of FC Barcelona vs Athletic Club.
The method is a rule-based multi-criteria score per frame (with AHP weights), smoothed by a Hidden Markov Model.
No training label is used anywhere: the only learned parts (the ball classifier and one team-shape model) are
trained on labels that an algorithm produces from the provided game itself. Everything is reproducible from the
raw assets with the scripts below. The method, the AHP judgements and the source of every constant are described in
[docs/technical_document.md](docs/technical_document.md).

## Method in one paragraph

1. **Ball height.** The ATD ball is the ball's position on the ground. The camera is recovered from the per-frame
   homography, so the vertical line above the ATD point can be projected into the image; whitish blobs near that line
   are scored by a gradient-boosting classifier and a Viterbi pass picks the ball and its height.
2. **Gap filling.** Where the ATD has no ball (33% of graded frames), a two-way tracker follows the ball through the
   video; short gaps (under 5 s) are interpolated. 67% of graded frames end up with a ball position.
3. **Criteria.** Per frame, dead-ball criteria D1-D3, D5-D7 and possession criteria T1-T6 (closeness, local dominance,
   last controller, Smart Tagging prior, team shape, motion match), with a control factor c that lowers the weight
   of closeness when the ball is in flight.
4. **Combination.** AHP weights (Saaty) and adaptive weighting give P(H), P(A), P(D) for each frame.
5. **HMM.** Viterbi decoding with an expected stay of 3 s for H and A and 10 s for D turns the flickering per-frame
   probabilities into stable possessions.

## Layout

```
scripts/     one script per pipeline stage, run in order (01 ... 09); the last one writes outputs/submission.csv
helpers/     the code the scripts use: one class per file
docs/        the technical document
outputs/     everything the scripts write (git-ignored; some files contain video frames: do not publish)
```

| Stage | Script | Writes | Class doing the work |
|---|---|---|---|
| 1 | `01_train_classifier.py` | `classifier.joblib` | `BallClassifier` (with `FrameReader`) |
| 2 | `02_detect_height.py`, `03_aggregate_height.py` | `blocks/`, `ball_height.csv` | `HeightDetector` |
| 3 | `04_fill_gaps.py`, `05_summarize_gaps.py` | `gaps/`, `gap_frames.csv` | `GapFiller` (with `BallTracker`) |
| 4 | `06_build_ball_track.py` | `ball_ground.csv` | `GapFiller` |
| 5 | `07_compute_criteria.py` | `criteria.csv.gz` | `Criteria` (with `Signals`, `Scoring`) |
| 6 | `08_combine_criteria.py` | `probabilities.csv.gz` | `AHP` |
| 7 | `09_smooth_labels.py` | **`submission.csv`** | `PossessionHMM` |

Other helper classes: `Settings` (paths and constants), `Match` (ATD tracking, homography, team slots, Smart Tagging),
`Camera` (camera model, pitch lines), `ParallelJobs` (worker pool).

## Running

```
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data          # the folder that contains tactical/ (or put it in a .env file)
python scripts/run_all.py --workers 6       # the whole pipeline -> outputs/submission.csv
```

The data folder can instead be set in a `.env` file in this folder: copy `.env.example` to `.env` and edit the path
(a variable already set in the environment takes precedence; `.env` is git-ignored).

`run_all.py` runs the numbered scripts in order (add `--from-stage N` to start at script N). They can also be run one by
one:

```
python scripts/01_train_classifier.py       # a few minutes
python scripts/02_detect_height.py --workers 6      # long (video), resumable per block
python scripts/03_aggregate_height.py
python scripts/04_fill_gaps.py --workers 6          # long (video), resumable per gap
python scripts/05_summarize_gaps.py
python scripts/06_build_ball_track.py
python scripts/07_compute_criteria.py       # about 5 minutes
python scripts/08_combine_criteria.py       # AHP weights -> P(H), P(A), P(D)
python scripts/09_smooth_labels.py          # HMM -> outputs/submission.csv
```

## Configuration

The code reads one setting from the environment and has no default for it: if it is missing the scripts stop with an
error message.

| Setting | Where | Meaning |
|---|---|---|
| `MTAH_DATA_DIR` | environment variable or `.env` file (see `.env.example`) | folder that contains `tactical/` with the five match files below. A variable already set in the environment takes precedence over `.env`. |

Files read from `<MTAH_DATA_DIR>/tactical/` (names are built in `helpers/settings.py`, class `Settings`, from the
constant `GAME = "BAR-ATH"`):

| File | Used by |
|---|---|
| `BAR-ATH_tactical.mp4` (the video) | scripts 01, 02 and 04 only |
| `BAR-ATH_tactical_atd.txt` and `_atd.xml` (tracking and team of each slot) | scripts 01-07 |
| `BAR-ATH_tactical_homography.csv` | scripts 01-07 |
| `BAR-ATH_tactical_smart_tagging.xml` | script 07 |

Everything the scripts write goes to `outputs/` next to the code (location fixed by `Settings.ROOT`, no setting needed).

**Parallel processes (`--workers`).** Scripts 02 and 04 read the video and are split into independent jobs (blocks of
100 frames, and gaps of the ball track). `--workers N` is the number of worker processes; the default is 6, also for
`run_all.py`, which passes it on. Choose a number up to the CPU cores of the machine. Each worker opens its own video
handle and loads the tracking arrays, so memory use grows with the number of workers; lower the number on a machine
with little memory. Results do not depend on the number of workers (every job is independent), and every finished job
is kept in `outputs/blocks/` or `outputs/gaps/`, so an interrupted run can be started again and continues where it
stopped. Rough cost, measured on the development machine: one block of 100 frames takes about 20-25 s in one process
and there are about 1,080 blocks, so script 02 needs roughly 6-7 hours of single-process time (about an hour with 6
workers); script 04 is also long (not timed in full); script 07 takes about 5 minutes and scripts 08-09 a few seconds. Actual times
depend on the hardware.

**Running without the video.** Scripts 07-09 read only the tracking, the homography and the Smart Tagging, plus two
files made by the video stages (`outputs/ball_height.csv` and `outputs/ball_ground.csv`). If those two files are
available the submission can be rebuilt without the video: `python scripts/run_all.py --from-stage 7` (the number
is the script number, here 07, the criteria).

Scripts 01-06 take the longest (01, 02 and 04 read the video); scripts 07-09 are fast and deterministic. A Poetry project is not needed:
the code is run as scripts and `requirements.txt` lists the five dependencies.

## Rules compliance

* No hand labels, and no corrected or human-produced data about this game, anywhere.
* The public ground truth (first 5 minutes) is used only to check the pipeline, never to train or tune. Every
  check made and the one revision that followed it are described in section 9 of docs/technical_document.md.
* The AHP judgements are written in `helpers/ahp.py` and justified in the technical document.
* The match assets are not stored in this repository; set `MTAH_DATA_DIR` to the downloaded folder.

## Results

| Check | Result |
|---|---|
| Ball height found on ATD frames (held-out windows) | 58% (61.1% over the whole match) |
| Graded frames with a ball position after gap filling | 67.0% (ATD alone: 59.3%) |

Dead-ball detection (D) is the weakest part of the method; see sections 9 and 11 of the technical document.
