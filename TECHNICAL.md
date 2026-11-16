# Frame-level ball possession with a restart-aware HMM

**More than a Hack 2027 — Phase 1** · Team **Blastoise**

## 1. Summary

We label every frame of the tactical video as home possession (`H`), away possession (`A`) or
dead ball (`D`). Frames outside the two graded halves are written as `X`. Labels come from a
hidden Markov model over {H, A, D}, decoded with Viterbi separately on each half. Two kinds of
per-frame evidence feed it, both computed from the automatic tracking after cleaning the ball
track:

- **ownership evidence** — whether the nearest player is close enough to control the ball and
  clearly closer than any opponent;
- **dead-ball evidence** — collective player speed, the ball crossing a line, a stationary ball
  with slow players, and a kicked ball.

The decoder encodes three rules of the game:

1. A change of possession needs sustained control.
2. A stoppage lasts at least 2.5 s.
3. After a stoppage, the team taking the restart has the ball first.

No model is trained and no labelled data is used. Every parameter is a physical or football
quantity, or a label-free measurement of the tracking itself.

On the full match the system reports **222 possessions** (median 11.6 s) and **96 stoppages**
(median 21.4 s). Of the graded frames, 33.4% are labelled `H`, 20.7% `A` and 45.9% `D`, which
corresponds to about 57 minutes of ball in play. Barcelona has 61.7% of live-ball frames. The
pipeline runs in about 5 seconds with numpy and pandas only, and reproduces byte for byte.

## 2. Why this approach

The ranking metric (`spellF1`) scores whole possessions, matched by temporal overlap. The
decisive property is therefore **segmentation**: producing the right number of possessions with
the right boundaries.

A per-frame classifier, however accurate, flickers and fragments. The published baselines show
this: a nearest-player rule produces several times too many possessions, and majority-vote
smoothing only partly repairs it.

An HMM states the trade-off directly. Every change of state has a cost, and Viterbi returns the
segmentation that best balances the evidence against the number of changes. It also lets us
build football's structure into the decoder itself, through a minimum stoppage duration and
restart rules, rather than patching the output afterwards.

## 3. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| Tactical ATD — players | **yes** | nearest-player distances; per-track speed for collective speed; restart positions (thrower beyond a touchline, taker at a corner flag) |
| Tactical ATD — ball | **yes** | cleaned into physical tracks (Section 5.1); position for ownership and lines; speed for kicks and flight |
| Tactical ATD metadata | **yes** | track-slot → team mapping (first `<Team>` is home) |
| Smart Tagging — phase tags | **yes**, weak prior | team-tagged attacking phases support that team's ownership; defending phases support the opponent's |
| Smart Tagging — set-piece team | **yes** | corners, goal kicks and free kicks name the restarting team; used as evidence for the restart team |
| Smart Tagging — kickoff tags | **yes** | locate the kickoffs opening each half and following each goal |
| Smart Tagging — goal tags | examined, **dropped** | the interval starts at the beginning of the attacking move, not at the goal; post-goal stoppages are already captured by slow players and an untracked ball |
| Smart Tagging — set-piece times | examined, **dropped** as timing | the real restart comes a variable number of seconds after the tag start, so the tags are used for *who* restarts, not *when* |
| Tactical homography / video | not used | the ATD is already in pitch coordinates |
| Broadcast video, ATD, homography | not used | not synchronised to the tactical timeline; alignment was out of scope |

## 4. Training data and use of the public ground truth

**No labelled data of any kind is used**: no human labels on the provided game or on any other
game. The few quantities measured from the data are label-free descriptive statistics of the
tracking itself (Section 5.2).

The public 5-minute ground truth was used only as the rules allow:

- reading it to understand what the labels mean (for example, that sub-second possessions exist
  and that stoppages coincide with players slowing down);
- running the official scorer as a pipeline correctness check, once per iteration.

It was **not** used to choose or tune any parameter, or to choose between iterations. For
transparency, the two checks gave the following. With 17 possessions the scorer reports gaps
below ~0.39 as noise, so these numbers carry no ranking information:

| | macroF1 | spellF1 | possessions (truth: 17) |
|---|---|---|---|
| iteration 1 | 0.769 | 0.493 | 13 |
| iteration 2 (submitted) | 0.810 | 0.463 | 15 |
| published floor (`baseline_smooth50`) | 0.621 | 0.326 | 26 |

The kickoff detector was checked against the two kickoff frames published in `FAQ.md` (11,185
and 113,742). Our detections are 11,188 and 113,744.

## 5. Method

### 5.1 Ball track cleaning

About 1.8% of consecutive ball detections imply speeds above 40 m/s, faster than any football.
These are tracking glitches. The cleaning works as follows:

- **Link into tracks.** Detections are linked into tracks whenever they are at most 8 frames
  apart and the implied speed stays at or below 40 m/s.
- **Drop blips.** Tracks with fewer than 4 detections are discarded.
- **Drop balls that never enter the pitch.** Tracks that never come at least 1 m inside the
  lines are discarded. These are spare balls held by ball boys, and 1,582 raw detections lay
  more than 5 m outside the pitch.
- **Interpolate short gaps.** Gaps inside a track are filled linearly.

Result: 113,488 raw detections become 110,607 kept detections on 297 tracks.

### 5.2 Label-free calibration of control distance

For slow balls on the pitch, at dribbling speed of 1.5–4 m/s, the nearest player is at a median
of 1.07 m (10th percentile 0.31 m). For balls faster than 8 m/s the median is 3.4–4.9 m: the ball
is in flight and nobody controls it. These measurements set the control radii and speed limits
below.

### 5.3 Ownership evidence (log-odds of H over A, per frame)

The evidence is `sign × w_dist × w_margin × w_speed`, where `sign` is +1 if the nearest player
is home and −1 if away. The three weights are:

- **`w_dist`** is 1 if the nearest player is within 1.0 m, falling linearly to 0 at 2.5 m.
- **`w_margin`** is 1 when the nearest opponent is at least 1.0 m further away. It falls
  linearly to 0 as the two become equally close, so a contested ball gives little evidence.
- **`w_speed`** is 1 for ball speeds up to 8 m/s, falling linearly to 0.3 at 12 m/s and above.
  This down-weights a ball in flight or a deflection.

Interpolated ball positions count half. When there is no ball there is no evidence, so possession
stays with the last team. A weak Smart Tagging phase prior of ±0.5 per second is added.

### 5.4 Dead-ball evidence (log-odds of D over live, per second)

- **Collective speed:** `2.0 × (1.3 − mean player speed)`, with speed smoothed over ±1 s and the
  result clipped to [−3, +1.5]. Before restarts players move at about 0.5–1.1 m/s, in play at
  about 2–3.5 m/s.
- **Ball beyond a line:** +25 while the ball is seen more than 0.5 m outside the pitch, but only
  on a track that came from the pitch, never on spare balls.
- **Ball lost leaving the pitch:** +2.5 from the moment the ball is lost until it is seen on the
  pitch again. This applies when the track's last detection is beyond a line, or when its last
  velocity would carry it across a line within 0.6 s. Seventeen tracks end this second way.
- **Stationary ball with slow players:** +1 while the ball moves under 0.5 m/s with collective
  speed under 1.3 m/s, which is a restart being set up.
- **Kicked ball:** −5 while a ball on the pitch moves faster than 8 m/s. A struck ball means play
  is live.
- **Ball untracked:** +0.5. This is weak, because the ball is lost often during play too.

### 5.5 Restart (taker) evidence

When the decoder leaves a stoppage, the cost of resuming with each team depends on who controls
the dead ball in the 2 s before the restart. The evidence for this comes from three sources:

1. The ownership evidence summed over that window, capped at ±10.
2. +3 per second during which only one team has a player standing where a restart is taken:
   more than 0.5 m beyond a touchline (a thrower) or within 2.5 m of a corner flag (a corner
   taker).
3. ±10 from a Smart Tagging set-piece tag naming the team, applied from 2 s before to 15 s after
   the tag start.

The total is capped at ±12.

The taker cap exceeds the switch penalty. So when the taker is known, the decoder prefers "restart
with the taker, then switch if the opponent gains control" over skipping the taker's possession.
This matches the challenge's rule that every restart has a clear possessor.

Two label-free checks support this choice:

- At throw-ins, the thrower's team agrees with the tracking-based restart team in 7 of 8 clear
  cases.
- At corners, a player at the flag always belongs to the tagged team (6 of 6).

### 5.6 Kickoffs

For each kickoff tag, the kick is the moment the tracked ball leaves the centre spot: the last
frame at its closest approach to the centre within the tag window. If the ball is not seen there,
the fallback is the first frame where collective speed crosses 1.3 m/s. Frames before the kick
are forced to `D`, from the start of the graded half for the half-opening kickoffs and from the
tag start after goals.

### 5.7 Decoding

**States.** H, A, and a chain D<sub>1</sub> … D<sub>63</sub>. A stoppage can only be left from D<sub>63</sub>, so it lasts at
least 63 frames (2.5 s).

**Emissions.**

- H and A: the favoured team scores 0 and the other scores −|ownership evidence|.
- D: the dead-ball evidence.

**Transition costs.**

- H ↔ A: 8. A change needs about 0.3 s of clear control by the other team.
- Live → D: 6.
- D → H or A: 6, plus the taker evidence against that team.

All scores are scaled by 1000 and rounded to integers, so Viterbi runs in exact integer
arithmetic. The result does not depend on floating-point summation order.

**Gap rule.** A switch that falls inside a stretch with no ownership evidence is moved to the
stretch's midpoint. Every position inside such a stretch scores the same, so this replaces an
arbitrary tie-break with the one that minimises the worst-case boundary error.

### 5.8 Development: label-free diagnostics that drove iteration 2

Iteration 1 was a plain three-state HMM without cleaning, minimum duration or restart logic. We
evaluated both iterations only against **ground-truth-free** signals: football base rates (a
match has roughly 100–120 restarts) and agreement with the automatic set-piece tags.

| diagnostic (full match) | iteration 1 | iteration 2 |
|---|---|---|
| stoppages | 160 | 96 |
| stoppages under 3 s (physically implausible) | 59 | 8 (all ≥ 2.5 s) |
| ...of which split one team's possession in two | 51 | 0 |
| open-play changes of possession | 70 | 127 |
| restart team agrees with set-piece tag | 10 / 22 | 20 / 22 |
| possessions | 228 | 222 |

The 59 short stoppages were a decoding loophole. Briefly entering and leaving `D` cost the same as
one team switch, so the decoder used fake stoppages to ride out contested moments instead of
assigning them to a team. The minimum-duration chain removes the loophole structurally.

### 5.9 Robustness

Changing each parameter by about ±25% and measuring the effect, with no ground truth involved:

| parameter | values | frames changed | possessions |
|---|---|---|---|
| `p_switch` | 6 / 10 | 2.5% / 1.9% | 268 / 197 |
| `p_to_dead`, `p_from_dead` | 4.5 / 7.5 | 1.8% / 0.7% | 230 / 223 |
| `s_mid` | 1.1 / 1.5 m/s | 6.6% / 4.9% | 224 / 216 |
| `min_dead_s` | 2.0 / 3.0 s | 0.0% / 0.0% | 222 / 222 |
| `r_far` | 2.0 / 3.0 m | 1.7% / 1.9% | 211 / 241 |
| `margin_full` | 0.75 / 1.25 m | 0.5% / 0.4% | 226 / 217 |
| `unseen_rate` | 0.25 / 0.75 | 1.6% / 1.8% | 222 / 224 |
| `tag_rate` | 0 / 1.0 | 3.8% / 2.1% | 227 / 226 |

The switch penalty and the control radius mainly set the number of possessions. Both are kept at
their physically motivated values in the middle of the tested range rather than chosen from any
score.

## 6. Possession rules implemented

- **Change of possession:** requires sustained control, roughly 0.3 s of a player within about
  1 m of a slow ball with the nearest opponent clearly further away.
- **Touches and deflections:** a single touch, a deflection, or a ball in fast flight past a
  player does not change possession.
- **Loose, contested or untracked ball:** possession stays with the last team.
- **Aerial duels:** a duel won without subsequent control does not change possession, because the
  ball's flight speed down-weights the evidence. There is no explicit aerial model.
- **Ball out of play:** `D` from the ball crossing a line (seen, or extrapolated from its last
  velocity), or from a sustained collective slowdown such as a foul, injury, offside or goal,
  until the restart.
- **Restarts:** the restarting team has the ball first. It is identified from the taker's control
  of the dead ball, the thrower or corner-taker position, and the set-piece tag.
- **Kickoffs:** `D` until the ball leaves the centre spot.
- **Minimum stoppage:** 2.5 s.
- **Dead-ball ownership:** not modelled. A dead ball is always `D`, as the challenge asks.

## 7. Limitations and what we would do with more time

- **Restart team for untagged restarts.** Where the ball is not tracked at the restart spot,
  restart-team evidence is weak. Tracking alone agrees with the set-piece tags only 11 of 20
  times without the tag evidence.
- **Quick restarts.** Throw-ins and goal kicks taken while players are still moving are sometimes
  missed. 44 of 59 tagged set pieces have a detected stoppage ending near the tag.
- **Slow live phases**, such as a goalkeeper holding the ball, can be mislabelled as dead.
- **Changes of possession while the ball is untracked** are timed at the midpoint of the gap,
  not observed.
- **Aerial balls.** The ball position is a ground projection, so high balls can pass close to
  players.
- **Hand-set parameters.** The number of possessions depends mostly on the switch penalty, which
  is set from football reasoning rather than learned.

Next steps we would take:

1. Learn the emission model from labelled tracking of other matches.
2. Use a semi-Markov model with realistic possession and stoppage duration distributions.
3. Detect restarts explicitly: a stationary ball followed by a kick.
4. Detect receptions from changes in ball velocity.
5. Align the broadcast tracking to fill tactical ball gaps.
6. Infer the restart team from pitch geometry and team attacking directions.

## 8. Reproduction

```bash
pip install -r requirements.txt                 # numpy==2.2.6, pandas==2.2.3 (Python 3.10–3.13)
# match data in ./data (or set MTAH_DATA_DIR); only the tactical ATD (.txt, .xml) and the
# Smart Tagging XML are read
python solution/run.py --team Blastoise       # -> submissions/Blastoise.csv, about 5 s
python evaluation/validate_submission.py submissions/Blastoise.csv
```

Expected SHA-256 of the output, verified on Linux (numpy 2.2.6 / pandas 2.2.3 and numpy 2.5.4 /
pandas 3.0.6) and on Windows with the pinned requirements:
`81b7485827192f2cad68af0b31d034b60cc7d48aee4fcc3f0102c29a728bc893`.
