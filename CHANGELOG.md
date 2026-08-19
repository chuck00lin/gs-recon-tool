# Changelog

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
