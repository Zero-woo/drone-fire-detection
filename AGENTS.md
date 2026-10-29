# Working rules
- University club project, initially CPU-only Windows. Explain results to the user in Korean.
- Install project dependencies and run project commands with the repository's .venv.
  On Windows, use .\.venv\Scripts\python.exe explicitly; do not install into system Python.
- Write every README.md file and all commit messages (subjects and bodies) in English.
- Keep the root README concise: project purpose and directory structure only.
  Subdirectory READMEs should briefly describe that directory's purpose and contents.
- Keep installation and usage instructions in docs/USAGE.md, progress updates in
  docs/PROGRESS.md, the experiment roadmap in experiments/ROADMAP.md, and detailed
  experiment results in reports/. Do not put progress logs or completion status in READMEs.
- Start with pretrained fire/smoke weights and saved images/videos.
- Training, live-camera capture, FC integration and physical actuation are later experiments.
- Run `python -m unittest discover -s tests -v` after experiment logic changes.
- Commit preparation, implementation and measured results as separate milestones.
- Never label synthetic tests or prepared code as a real model experiment.
- Do not commit raw media, weights, environments, credentials or full run directories.
- Record model/input SHA256, effective settings, software versions and Git state.
- Use immutable model revisions; fail visibly on checksum mismatches.
- Compare on the same held-out recordings; split by video/session before frame extraction.
- The user authorized ordinary milestone commits/pushes to
  https://github.com/Zero-woo/drone-fire-detection.git. Never force-push.
- Use work/ for scratch, experiments/runs/ for generated runs, outputs/ for exports.
