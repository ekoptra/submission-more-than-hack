# Technical document: ball possession (H / A / D) from tactical-video tracking

Team PVD. More Than A Hack 2027, phase 1. Match: FC Barcelona vs Athletic Club (tactical video, 199,319 frames at 25 fps,
158,857 graded frames). This document describes what we built, how each part works, which numbers it rests on, how we
checked it, and where it fails.

## 0. At a glance

| | |
|---|---|
| Method | rule-based multi-criteria score per frame (AHP weights) smoothed by a Hidden Markov Model (HMM); no possession model is trained |
| Learned components | a ball-candidate classifier and a team-shape model, both trained only on labels that an algorithm produces from this match |
| Hand labels, external labelled data, ground truth in training or tuning | none |
| Ball position coverage on graded frames | 67.0% (ATD alone: 59.3%), including height on 61.1% of the frames the ATD covers |
| Criteria | 6 dead-ball (D) and 6 possession (T), each with a written source for every constant |
| Public 5-minute window | used only to check that the pipeline is correct (alignment, file format, labels meaning what we think); never to train or tune (section 8) |
| Weakest part | dead-ball detection (section 11) |
| Code | one class per file in `helpers/`, nine numbered stage scripts, one command regenerates the file |

## 1. The problem as we read it, and what the data allows

The task asks for one label per frame, but the ranking metric (spellF1) scores whole possessions as time intervals. Two
consequences shaped everything: a per-frame label that flickers is worthless (the organisers' own raw baseline invents several times
more possessions than a match contains), and a dead ball is a stoppage, so `H D H` is two
possessions. We therefore treat the problem in two layers: evidence per frame, then a sequence model that decides
where possessions start and end.

Facts about the data that we measured or took from the documentation and designed around:

* The ATD ball is missing on **40.7%** of graded frames (64,643 frames in 150 gaps; the median gap lasts 10.7 s and the
  longest 221.5 s). Most missing frames are in long gaps, not in one-frame flickers.
* The ATD ball position is the ball **projected on the ground**. An airborne ball is recorded under where it is, so its
  distance to players is wrong exactly when a pass is in flight.
* The ATD has 46 track slots for 22 players; a slot is a tracklet, not a person, and the team of a slot is correct only
  about 92-94% of the time. The ATD speed column is empty on about 8% of observations, so we compute speeds from positions.
* Smart Tagging is automatic, not ground truth: 59 set-piece phases (53 of fixed length 35 s, 6 kick-offs of 5 s), and
  every throw-in is tagged `Team = N/A`.
* Median player speed during the match is already about 1.3 m/s, and about the same inside the Smart Tagging set-piece
  windows (1.45 m/s), so **collective speed alone does not separate a stoppage from slow play**. We found this early and
  it explains most of the difficulty with the dead-ball class.

## 2. Assets used

| Asset | Use |
|---|---|
| Tactical video | finding the ball (candidates, height, gap tracking); the criteria do not read it directly |
| ATD tracking (46 slots and ball) | player positions and speeds, team of each slot (from the metadata), ball ground position |
| Tactical homography | camera recovery and the mapping between pitch and image |
| Smart Tagging | weak prior only: the set-piece phase (D6) and the team tag of the running phase (T4) |

Not used: the broadcast video, tracking and homography (they are not synchronised with the tactical video, and mapping them back
would have required a method of its own), and the ATD speed column (empty on about 8% of observations; speeds are computed from
positions). The public 5-minute ground truth is not an input of any stage.

## 3. Pipeline

| Stage | Script | What it does | Written to |
|---|---|---|---|
| 1 | `01_train_classifier.py` | trains the ball-candidate classifier | `classifier.joblib` |
| 2 | `02_detect_height.py`, `03_aggregate_height.py` | ball height on frames that have an ATD ball | `ball_height.csv` |
| 3 | `04_fill_gaps.py`, `05_summarize_gaps.py` | follows the ball through gaps | `gap_frames.csv` |
| 4 | `06_build_ball_track.py` | one ball position per graded frame, with its source | `ball_ground.csv` |
| 5 | `07_compute_criteria.py` | per-frame criteria matrix | `criteria.csv.gz` |
| 6 | `08_combine_criteria.py` | AHP weights -> P(H), P(A), P(D) | `probabilities.csv.gz` |
| 7 | `09_smooth_labels.py` | HMM -> labels | `submission.csv` |

Stages 1-4 are preprocessing: they only improve the quality and coverage of the ball data and decide nothing about
possession. Stages 5-7 are the possession logic and read only tracking, homography and Smart Tagging plus two files from
the video stages.

## 4. Recovering the ball from the video (stages 1-4)

### 4.1 Camera from the homography

The per-frame homography maps pitch to image only on the ground plane. Assuming square pixels, no lens distortion and the
principal point at the image centre (960, 540), the focal length and camera pose can be recovered from each homography, which
lets us project a 3-D point (x, y, z) into the image. For an ATD point (x, y) we project the vertical line
(x, y, z) for z from -0.5 m to 8 m in 0.05 m steps. The airborne ball must lie near that curve; the slightly negative lower
bound tolerates small homography and ATD errors.

### 4.2 Ball candidates

On every frame with an ATD ball we look for small whitish blobs inside the pitch polygon (grown by 4 m), with pitch
lines masked out (9 px wide). A pixel is whitish if V > 115 and S < 110 in HSV and it is not grass. Connected blobs are kept
as candidates if area is 4-220 px, longest side at most 26 px, aspect ratio at most 3.5 and fill ratio at least 0.30 (at most
150 per frame). This looser set raised the share of ATD-ball frames where a ball was later found from 39% to 47% in our tests.

We deliberately do **not** erase players from the image: a goalkeeper may hold the ball and boots can be white. Every blob
stays a candidate and is described by 13 features, among them size and shape, brightness and contrast to a surrounding ring,
whether the blob lies inside the projected body box of a player (0.9 x 0.9 x 1.9 m) and its distance to the nearest foot. The
classifier learns to distrust boots without our deleting any player.

### 4.3 Classifier and its labels (no hand labels)

A gradient-boosting classifier (200 stages, learning rate 0.08, balanced class weights, seed 0) scores each candidate.
Its labels are generated from the ATD by the algorithm itself:

1. 12 windows of 100 frames are chosen deterministically (seed 11) where the ATD ball is present on at least 95% of frames, away
   from the windows we keep aside for checks.
2. **Stage 1.** A candidate within 10 px of the ATD ground pixel is a ball; one more than 40 px away and outside the band above
   the point where an airborne ball could be (25 px wide, up to 130 px high) is not. Result: 42,335 candidates, 245 positives.
3. **Stage 2 (self-training).** Stage 1 only teaches balls on the grass, so the stage-1 model is run through the height
   detector below, and its detections within 8 px of the vertical curve are added as positives (airborne balls), with
   unselected candidates more than 40 px from the curve as negatives. Result: 84,306 candidates, 565 positives (320 from
   detections).

Retraining from scratch reproduces the saved model exactly (probability difference 0.0 on 506 test candidates).

### 4.4 Ball height

Candidates within 30 px of the vertical curve are scored as ln(p + 0.001) - 0.5 (d / 10)^2, with p the classifier score and
d the distance to the curve. The choice is not made frame by frame, because a single frame can mislead (a boot that looks like a
ball). Instead a Viterbi pass over a block of 100 consecutive frames picks the best sequence, in which each frame either
chooses one candidate or declares "ball not found" (cost ln 0.08) and a change of height between two chosen candidates costs
min(0.5 (dz / 0.8 m)^2, 4). The height is read off the curve at the point closest to the chosen candidate. Blocks are read with 10
extra frames so the sequence is not cut at a block edge.

We tested and kept the 30 px band: wider bands admit boots and bodies, and the remaining failures (ball far from the ATD point,
or hidden in the net or behind a player) cannot be fixed that way. Coverage rose step by step on held-out windows (599 ATD-ball
frames): 39% (strict filter, 12 px band) -> 47% (looser filter) -> 56% (30 px band) -> 58% (retrained classifier). Over the whole
match the height is found on **57,564 of 94,214** ATD-ball frames (61.1%); half of the 1,076 blocks reach 64% or more and 81 blocks
stay below 20%. Heights have no reference, so the only check is plausibility: between consecutive frames the height changes by
at most 5 cm in half the pairs and at most 15 cm in 90% (52,820 pairs).

### 4.5 Filling the gaps

* **Gaps up to 5 s (125 frames, 18 gaps, 197 frames):** straight line between the ATD ground positions at both ends.
* **Longer gaps:** a two-way tracker follows the ball in the video, forward from the last detection before the gap and
  backward from the first detection after it. State: pixel position, pixel velocity and the last known height. Each frame it
  predicts the next position, accepts only candidates inside a gate that widens with every missed frame
  (radius = min(160, 30 + 12 m) px, m = consecutive misses), ignores candidates with classifier score below 0.15, accepts a
  candidate on another player's body only if its score is at least 0.5 (or it is the player who had the ball), and gives up after
  25 consecutive misses. The ground position of a pick comes from inverting the camera at the last known height.
* **Acceptance policy:** the two trackers must agree within 5 m to be averaged (`agree`); a single answer is used as is
  (`one`); two answers that disagree are discarded.

| Source of the ball position on the 158,857 graded frames | Frames | Share |
|---|---|---|
| ATD | 94,214 | 59.3% |
| short gap, straight line | 197 | 0.1% |
| long gap, trackers agree | 1,241 | 0.8% |
| long gap, one tracker | 10,825 | 6.8% |
| none | 52,380 | 33.0% |

Long gaps are hard: for gaps of 20 s or more the trackers answer on only about 15% of frames. Velocity is never computed from
tracker positions, only from ATD or short-gap positions.

## 5. Criteria (stage 5)

Every criterion turns a measurement into a score in [0, 1] with `ramp(x; a, b) = clip((x - a) / (b - a), 0, 1)` (0 at x = a, 1 at
x = b; decreasing when b < a). Windows (+-N s) average a score over time so one noisy frame cannot trigger it.

### 5.1 Dead-ball criteria (D)

| Code | Criterion | Score | Source of the constants |
|---|---|---|---|
| D1 | collective speed | ramp(median player speed, +-2 s; 1.33 -> 0.53 m/s) | percentiles 50 and 10 of this match, unlabelled |
| D2 | ball beyond the pitch lines | ramp(distance outside, +-1 s; 0.5 -> 2.0 m) x ramp(ball speed; 3 -> 0.5 m/s) | game laws; ATD position error; ball slow |
| D3 | players standing still | ramp(share of players slower than 0.5 m/s; 0.14 -> 0.50) | percentiles 50 and 90 of this match, unlabelled |
| D5 | ball untracked | share of frames in +-1 s without any ball position | no measured scale: weak evidence, so it has the lowest weight |
| D6 | Smart Tagging set-piece phase | 1 inside a SET PIECES phase | automatic annotation, weak prior |
| D7 | lone still player at a touchline or corner flag | max over players of ramp(speed; 0.5 -> 0.1) x ramp(distance to nearest player; 2 -> 4 m) x ramp(distance to line or flag; 2.0 -> 0.5 m), +-1 s | game laws and gait physics |

D2 is deliberately conservative: a ball that only touches or runs along a line must not count as out of play, so the score
starts 0.5 m beyond the line (ATD error and line width), needs the ball to be slow and is averaged over 2 s.

### 5.2 Possession criteria (T) and the control factor

| Code | Criterion | Score (0.5 means no evidence, 1 means home) |
|---|---|---|
| T1 | closeness | 0.5 + c (dA / (dH + dA) - 0.5), with dH and dA the distances from the ball to the nearest home and away player |
| T2 | local dominance | 0.5 + c ((nH + 0.5) / (nH + nA + 1) - 0.5), with nH and nA the players within 3 m of the ball |
| T3 | last controller | 0.5 + 0.5 s max(0, 1 - dt / 3), s = +1 or -1 for the team that last had c > 0.6, dt the time since (not defined after 3 s) |
| T4 | Smart Tagging team | 1 / 0 for the team tag of the running phase, 0.5 for N/A |
| T5 | team shape | logistic model on centroid, depth and width of both teams (section 5.4) |
| T6 | motion match | 0.5 + 0.5 e (mH - mA), with m the best ramp(velocity-vector difference to the ball; 4 -> 1 m/s) among players within 3 m, and e = ramp(ball speed; 0.5 -> 2 m/s) |

The **control factor** `c = c_distance x c_speed x c_height` says how much a frame can be trusted as evidence of who has the ball:
`c_distance = ramp(nearest-player distance; 3 -> 1 m)`; `c_speed = ramp(ball speed relative to the nearest player; 8 -> 2 m/s)` (a
carried ball moves with its carrier, so its relative speed is small even when it is fast; a pass moves away from everyone);
`c_height = ramp(height; 3 -> 1.5 m)`; unknown speed or height counts as 0.7. We first used absolute ball speed and found that it
kept `c` low on most frames (only 19.4% of ball-known frames had c > 0.6, because carrying a ball already means 5 m/s); with relative
speed the share rose to 27.8%.

### 5.3 Combination and adaptive weights

The decision is made in two stages, dead ball first and then team:

```
s_D = sum(w_k D_k) / sum(w_k)               (over the D criteria computable on the frame)
q_H = sum(w_k a_k T_k) / sum(w_k a_k)       (a = c for T1, T2; a = e for T6; a = 1 otherwise)
P(D) = s_D,   P(H) = (1 - s_D) q_H,   P(A) = (1 - s_D) (1 - q_H)
```

The T weight set depends on whether the ball position is known on the frame (T1 .. T6) or not (T3, T4, T5 only). The
**adaptive weights** matter in flight: with plain weights, T1 and T2 (together about half of the weight) stay neutral at 0.5 while the
ball is in the air and dilute T3: with T3 = 1 for a team that has just passed the ball and every other criterion neutral, q_H is only
about 0.57. Scaling by `c` lets T3 decide.

### 5.4 Team-shape model (T5) and leakage control

A logistic regression on nine features (centroid along the attack direction, depth and width of each team, and their
differences; the rearmost player is left out as the goalkeeper; the attack direction of each half is found from the sides the
teams occupy at kick-off). Its label is the team of the player nearest to the ATD ball on frames where the ATD has the ball. To avoid
predicting a frame with a model that has seen it, the match is cut into 5-minute blocks and each block is predicted by a model
trained on the others (20% of training frames, `C = 1`, seed 0). It agrees with its automatic label on 67.0% of frames, which makes it a
weak clue, so it has a small weight when the ball is known (0.061) and a large one when it is not (0.444), where little else exists.

## 6. Weights by the Analytic Hierarchy Process (stage 6)

For each criterion group we wrote a pairwise comparison matrix (Saaty scale 1-9, entry a_ij = how many times more important
criterion i is than j), took the principal eigenvector as the weights and checked the consistency ratio CR (acceptable below 0.10).
Each comparison weighs three dimensions: discriminating power according to the laws of the game, reliability given ATD and
tracking errors, and coverage (on how many frames the criterion exists). The judgements are written in `helpers/ahp.py` and were
fixed before any evaluation.

| Group | Consistency ratio | Weights used |
|---|---|---|
| Dead ball (D1-D7, then D4 removed, see section 9) | 0.016 | D1 0.384, D2 0.244, D3 0.146, D6 0.092, D7 0.092, D5 0.042 |
| Team, ball known (T1-T6) | 0.015 | T1 0.376, T6 0.236, T2 0.143, T3 0.143, T5 0.061, T4 0.040 |
| Team, ball unknown (T3, T4, T5) | about 0 | T3 0.444, T5 0.444, T4 0.111 |

Sensitivity: each of the 15 weights moved by -20% and +20% (the others rescaled to sum to one) changes at most **0.88%** of the
final labels (median 0.33%) and gives between 371 and 387 possession segments (final: 382).

## 7. Hidden Markov Model (stage 7)

States H, A, D per half (halftime is not graded). The emission log-likelihood of a frame is ln max(P(state), 0.001) from stage 6. A
state is kept with probability 1 - 1/(25 T), T being its expected stay in seconds, and the rest is split over the other states:

| From | T | Stay | To each other state |
|---|---|---|---|
| H, A | 3 s | 0.9733 | other team 1/150, to D 1/75 |
| D | 10 s | 0.9960 | to H and to A 1/500 each |

T = 3 s for H and A comes from the challenge definition ("a player keeps the ball for a few seconds"); T = 10 s for D from the fact that
throw-ins, goal kicks and injuries last several seconds to tens of seconds. The most probable state path is found with the Viterbi
algorithm, which turns the flickering frame-by-frame labels into stable possessions. On the
whole match the output has 382 possession segments (mean 13.1 s, median 8.5 s) and 90 dead-ball segments (mean 15.1 s), with 45.8% H,
32.8% A and 21.4% D.

## 8. Possession rules we implemented

* A dead ball is a stoppage, never a possession; there is no team on `D`.
* A loose or contested ball stays with the team that last clearly controlled it, for 3 s through criterion T3 and then through the HMM,
  which keeps the previous label until new evidence arrives. While the ball is in flight the closeness evidence is down-weighted by
  the control factor and the adaptive weights, so it cannot move the label.
* A challenge or loose ball changes possession only when the other team clearly controls the ball (T1, T2, T6 with high `c`).
* Restarts are recognised as dead-ball evidence (ball beyond the lines, a lone player at the line, the Smart Tagging set-piece window).
  We never write `X` on a graded frame; it appears only outside the two graded halves.

## 9. Integrity and checks

**Rules compliance.** No hand labels, no human-produced data about the match, no external labelled data. The two learned components use
labels produced by algorithms from this match (sections 4.3 and 5.4). The AHP judgements are written expert opinion (`helpers/ahp.py`),
fixed before any evaluation, and every constant has a stated source (game laws, physics, percentiles of the unlabelled tracking,
explicit expert judgement, or the challenge definition).

**Use of the public 5-minute window.** It was used only to check that the pipeline is correct, on three occasions, and no
parameter was chosen by maximising a score on it (the window holds only 17 possessions, which cannot rank configurations). For
transparency we disclose one decision taken after seeing results from it: after the second check we removed criterion D4 ("ball almost
still", multiplied by D1) and lengthened the expected stay in the dead-ball state from 3 s to 10 s. We made both changes for physical
reasons (during a stoppage the ball is carried to the restart point and players walk; stoppages last longer than a short
possession), and the third check was not used to select any other value.

**Checks that do not need labels.** (a) Criteria coverage and distribution over all 158,857 graded frames (for example D1 exceeds 0.5 on 23.2% of
frames, D2 on 1.4%, D7 on 4.1%); (b) rank correlation between criteria, to see which ones repeat each other; (c) the share of dead ball
(21.4%) and the number and length of possessions (above) for plausibility; (d) the weight sensitivity of section 6; (e) visual
inspection of random samples of ball detections and of the frames that trigger criterion D7 (about a third of 12 sampled
frames showed a substitution, an official or staff rather than a throw-in taker, which is why D7 has only a moderate weight); (f)
rejecting a walking-speed version of D1 and D3 because it labelled 67% of the whole match as dead ball.

## 10. Reproducibility and engineering

* One command regenerates the file (`python scripts/run_all.py`); the data folder is set once in `.env` (`MTAH_DATA_DIR`) and there is
  no default path in the code. Stages 2 and 4 run in parallel worker processes and are resumable (every finished block or gap is kept).
* Determinism: fixed seeds (11 for window selection, 0 for the models) and no unseeded randomness. After refactoring the code to
  one class per file we re-ran stages 03 to 09 and obtained **byte-identical** ball tables and final labels, a criteria matrix identical
  to the previous one, and identical results on samples of stages 01, 02 and 04.
* Timing on our machine: about 20-25 s per 100-frame block in one process, with about 1,080 blocks (an hour or so with 6 workers for the
  height stage, estimated from single blocks); the criteria stage about 5 minutes; combination and smoothing a few seconds. Stages 5-7
  need only tracking, homography, Smart Tagging and two files from the video stages, so the final labels can be rebuilt without the video.
* Dependencies: Python 3.12.2 and five packages pinned to exact versions in `requirements.txt` (numpy, pandas, OpenCV, scikit-learn, joblib). Video decoding and numerics can
  differ very slightly across machines or library versions; the byte-identical checks above were made only on our machine.

## 11. Limitations and what we would do with more time

* **Dead ball is the weakest class**: it is detected on a minority of the true stoppage frames. Collective speed does not separate stoppages from slow play and the specific
  criteria (D2, D7) fire on few frames (1.4% and 4.1%). We would build a restart detector (the kick or throw that ends a stoppage and the
  ball leaving the lines that starts one) and tell officials and substitutes from players, which the ATD cannot do.
* 33.0% of graded frames have no ball position, so there the label rests on indirect criteria (team shape, last controller, Smart Tagging).
* Ball height has no reference, and the accuracy of positions from the long-gap tracker was measured on simulated gaps during development
  but not on the final code, so we do not quote a figure.
* Team per slot is 92-94% correct; the criteria are correlated (Spearman 0.90 between T1 and T2, 0.87 between T1 and T3), so some evidence
  is counted more than once; sensitivity to the score thresholds was not measured and the AHP judgements are subjective.
* The constants marked as physics or game knowledge are approximations without literature citations.

## 12. Reproduction

```
pip install -r requirements.txt
cp .env.example .env          # set MTAH_DATA_DIR to the folder that contains tactical/
python scripts/run_all.py --workers 6      # writes outputs/submission.csv
```

Repository layout: `scripts/` (stages 01-09 and `run_all.py`), `helpers/` (one class per file: `Settings`, `Match`, `Camera`, `FrameReader`,
`BallClassifier`, `HeightDetector`, `BallTracker`, `GapFiller`, `Scoring`, `Signals`, `Criteria`, `AHP`, `PossessionHMM`, `ParallelJobs`), `docs/`
(this document). The match data is not part of the repository.
