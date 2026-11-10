# Models

Checksummed checkpoints and selected model revisions used by the project.

- `coco/`: COCO-pretrained YOLOv8n initialization for independent training runs.
- `published/`: the published D-Fire fire/smoke reference model.
- `frozen/`: selected trained models, fixed inference settings, and provenance records.
- Pretrained checkpoint sources and checksums are defined in `configs/models.json`.
- Frozen revisions carry their own `model_revision.json` and `SHA256SUMS`.
