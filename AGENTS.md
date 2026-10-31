# Working rules
- University club project, Windows with an NVIDIA RTX 5070. Explain results to the user in Korean.
- Install project dependencies and run project commands with the repository's .venv.
  On Windows, use .\.venv\Scripts\python.exe explicitly; do not install into system Python.
- Write every README.md file and all commit messages (subjects and bodies) in English.
- Keep the root README concise: project purpose and directory structure only.
  Subdirectory READMEs should briefly describe that directory's purpose and contents.
- Keep installation and usage instructions in docs/USAGE.md, progress updates in
  docs/PROGRESS.md, the experiment roadmap in experiments/ROADMAP.md, and detailed
  experiment results in reports/. Do not put progress logs or completion status in READMEs.
- Fine-tune COCO-pretrained YOLOv8n on D-Fire; compare hyperparameters on validation data.
- Start each independent training experiment from the pinned COCO yolov8n.pt checkpoint.
- Keep published fire/smoke weights as a reference; published scores are not local results.
- Live-camera capture, FC integration and physical actuation are outside the current scope.
- Run `python -m unittest discover -s tests -v` after experiment logic changes.
- Commit preparation, implementation and measured results as separate milestones.
- Never label synthetic tests or prepared code as a real model experiment.
- Do not commit raw media, weights, environments, credentials or full run directories.
- Record model/input SHA256, effective settings, software versions and Git state.
- Use immutable model revisions; fail visibly on checksum mismatches.
- Keep train/validation/test separate; evaluate test only after freezing model and thresholds.
- Use supplied official split lists; document differences from the published model's split.
- For new video data, split by video/session before frame extraction. Document unavailable source grouping.
- The user authorized ordinary milestone commits/pushes to
  https://github.com/Zero-woo/drone-fire-detection.git. Never force-push.
- Use work/ for scratch, experiments/runs/ for generated runs, outputs/ for exports.
