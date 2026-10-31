# Drone Fire Detection

Reproducible YOLOv8n fine-tuning on D-Fire to study the effect of training
hyperparameters on fire and smoke detection with an NVIDIA RTX 5070.

## Project Structure

```text
configs/                Dataset, model provenance, and staged experiment settings
src/fire_detection/     Dataset audit, training, evaluation, and analysis
firelab.py              Command-line entry point
scripts/                Windows CUDA environment setup
notebooks/              Colab training notebook
tests/                  Automated checks
data/raw/               Unmodified D-Fire data and official split lists
data/processed/         Validated training copies, manifests, and data.yaml
models/                 Downloaded model checkpoints
experiments/            Experiment roadmap
experiments/runs/       Generated results for individual runs
reports/                Reviewed validation and experiment reports
docs/                   Usage, project plan, and progress notes
outputs/                Exported results for sharing
work/                   Temporary files
```
