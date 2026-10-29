# Drone Fire Detection

CPU-first fire and smoke detection for a university drone project. The software
uses pretrained models to analyze images and recorded videos, with reproducible
settings and experiment records. The broader project aims to connect visual
fire detection with drone navigation and water-balloon release.

## Project Structure

```text
configs/                Model provenance and experiment settings
src/fire_detection/     Model preparation, media input, and inference
firelab.py              Command-line entry point
scripts/                Environment setup and milestone commits
tests/                  Automated checks
data/raw/               Original images and videos
data/processed/         Extracted frames and processed data
data/manifests/         Data sources and recording conditions
models/                 Downloaded model checkpoints
experiments/            Experiment plans and roadmap
experiments/runs/       Generated results for individual runs
reports/                Reviewed experiment reports
docs/                   Usage instructions and progress notes
outputs/                Exported results for sharing
work/                   Temporary files
```

Raw media, model weights, environments, and generated run files are excluded
from Git.
