# More than a Hack 2027 — Phase 1 submission: frame-level ball possession

Team: **Blastoise**

For every frame of the tactical video this repository outputs `H`, `A`, `D` or `X`, using a
hidden Markov model over home possession, away possession and dead ball, decoded with Viterbi
over evidence from the automatic tracking (after cleaning the ball track). The decoder enforces
three rules of the game: a change of possession needs sustained control, a stoppage lasts at
least 2.5 s, and the team taking a restart has the ball first. The method,
the assets used and the possession rules are described in **[TECHNICAL.md](TECHNICAL.md)**.

This repository is a fork of the official challenge repository
([metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027),
MIT licence). Its original README is kept as [CHALLENGE_README.md](CHALLENGE_README.md). Our code is
in `solution/`; the scorer, validator and loaders are unchanged.

## Reproduce the submission

Requires Python 3.10–3.13 (tested on 3.12) and the match data bundle, which is **not** in this
repository.

```bash
# from the folder of this repository (git clone, or Code -> Download ZIP on GitHub)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # numpy and pandas, pinned

# Put the match data in ./data (tactical/ and broadcast/ subfolders, original file names),
# or point MTAH_DATA_DIR at it.

python solution/run.py --team Blastoise               # -> submissions/Blastoise.csv (~5 s)
python evaluation/validate_submission.py submissions/Blastoise.csv
```

Only these files are read: `data/tactical/BAR-ATH_tactical_atd.txt`,
`data/tactical/BAR-ATH_tactical_atd.xml` and `data/tactical/BAR-ATH_tactical_smart_tagging.xml`.

The pipeline has no randomness and decodes on integer scores, so the output is identical across
machines; it has been verified byte for byte on Linux and on Windows. SHA-256 of the submitted
file:

```
81b7485827192f2cad68af0b31d034b60cc7d48aee4fcc3f0102c29a728bc893
```

(`sha256sum submissions/Blastoise.csv` on Linux/macOS,
`Get-FileHash submissions\Blastoise.csv` in PowerShell.)

## Layout

```
solution/possession.py   ball cleaning, features, Smart Tagging priors, kickoffs, restart-aware Viterbi
solution/run.py          command-line entry point that writes the submission
TECHNICAL.md             the technical document
```

No match data, video or frame-level derived data is committed; `data/` and `submissions/` are
gitignored.
