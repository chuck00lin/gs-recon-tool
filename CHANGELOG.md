# Changelog

## 1.1.0

Frame extraction correctness, and a Stop button that stops.

### Fixed

- **Re-extracting into an existing project no longer mixes two captures.** Extraction
  overwrote `frame_0000` onwards and left everything above it from the previous run, so a
  second pass at a lower sample rate produced a folder whose filename order jumped backwards
  partway through — precisely the order COLMAP's sequential matcher trusts to be capture
  order. Frames from an earlier extraction are now removed first; `--keep-existing` opts out.
- **Frame names no longer sort wrongly past 9999.** `frame_10000.jpg` sorted between
  `frame_1000` and `frame_1001`, scrambling every consumer downstream for any run longer than
  11 minutes at 15 fps. Names are now zero-padded to six digits.
- **The sample rate is the rate you get.** Sampling used an integer frame stride, so 20 fps
  out of a 29.97 fps clip rounded to "every frame" (50% more frames than asked for) while the
  same setting on a 30.00 fps clip gave 15 fps. Frames are now selected against a timeline,
  which also keeps the spacing even on variable-frame-rate phone footage, where a fixed
  stride made the real interval swing by 5x.
- **Stop actually stops.** `docker run` forwards signals into the container, where PID 1 —
  an entrypoint script or `bash -lc` — ignores SIGTERM by kernel rule, so training ran on
  with the pipeline stuck reading its output. Containers now start with `--init`, and every
  `docker run` is named so it can be killed outright if the signal is ignored anyway.
  Ctrl-C does the same thing on the CLI instead of unwinding Python around a live container.
- The GUI batches log output instead of emitting one signal and one widget update per line.
  A training run prints fast enough to saturate the event loop, which is what made the Stop
  button unclickable in the first place.
- Extraction warns instead of stopping silently when decoding ends early, and reports the
  effective frame rate it achieved.

### Changed

- `frames.filter.scalar` became `frames.filter.group_size`: how many consecutive frames
  compete against each other, with the sharpest of each group surviving. The old spelling was
  a power-of-two divisor on the group *count* — two indirections away from anything visible.
  Existing configs are migrated on load, so `scalar: 2` at `target: 20%` becomes
  `group_size: 10` (the sharpest 2 of every 10).
- The GUI computes the selection live, from the real length of the loaded clips: how many
  frames extraction will produce, how they divide into groups, how many survive per group,
  and a suggested group size when the current one would clump or starve the selection.
- The `Fast draft` and `High quality` presets set a group size matched to their keep rate.

## 1.0.0

First standalone release, extracted from `pipeline_assist_gui.py` in the internal
`3dplant-workflow` repository.

### Added

- `gs-recon` CLI: `doctor`, `setup`, `init`, `plan`, `run`, `gui`, plus the `frames` and
  `prune` tools as subcommands.
- `gs-recon doctor` — environment self-check where every failure carries the command that
  fixes it, including verification that the LichtFeld mount path matches the RUNPATH baked
  into the binary.
- YAML configuration shared by the GUI and CLI, so a run designed in the window can be saved
  and replayed headlessly. `--set key=value` overrides single values.
- `gs-recon plan` prints every command before anything runs.
- Resume from the failing step, in both front ends.
- Redesigned GUI: project tree with live per-step status on the left, one tab per stage on
  the right, Docker settings quarantined in *Advanced*, draggable log, progress bar,
  quality presets, and an in-window environment check.

### Changed

- **LichtFeld now runs via `docker run --rm` with the dataset bind-mounted at its own path**,
  replacing `docker exec` into a long-lived container. The old approach only worked when the
  dataset happened to sit inside a path that container already mounted, which is why the
  tool did not travel between machines.
- Hard-coded paths (`/home/chucklab/projects/LichtFeld-Studio`, container names, the
  vocabulary tree location) became configuration with autodetection.
- The vocabulary tree moved out of the repository to `~/.local/share/gs-recon/` and is
  fetched by `gs-recon setup --assets`.
- Command construction was separated from the Qt widgets, which is what made a CLI possible
  and what guarantees the two front ends emit identical commands.
- `graphlib.py` renamed to `ascii_graph.py`: the old name shadowed Python 3.9+'s standard
  library module whenever its directory landed on `sys.path`.

### Removed

- The LichtFeld version switcher. Releases pin GUT mode; the old build's `--pose-opt` path no
  longer exists upstream, and the switch forced every user to know which image they had.
- `--gpu N` on the training command. That flag does not exist in current LichtFeld Studio;
  GPU selection now goes through `docker --gpus` where it belongs.
- The `{mem_available_gb}` placeholder expansion in extra arguments, which existed only for
  the old Docker image's free-memory argument.
