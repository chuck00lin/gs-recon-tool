# Configuration reference

One YAML file describes an entire run. The GUI reads and writes it; the CLI reads it.

```bash
gs-recon init                        # ~/.config/gs-recon/config.yaml
gs-recon init -o project.yaml        # somewhere specific
gs-recon init --stdout               # just print it
gs-recon run video.mp4 -c project.yaml
```

Resolution order: `-c FILE` if given, else `~/.config/gs-recon/config.yaml` if it exists,
else the built-in defaults. Override single values without editing the file:

```bash
gs-recon run video.mp4 --set splat.iterations=15000 --set frames.filter.target=30%
```

Unknown keys are an error rather than a silent no-op, so a typo surfaces immediately.

---

## `docker`

| Key | Default | Notes |
|---|---|---|
| `colmap_image` | `jinwj1996/glomap` | Public image holding both COLMAP and GLOMAP |
| `lfs_image` | `lichtfeld-studio:latest` | Built locally; cannot be pulled |
| `lfs_repo_host` | *(autodetected)* | Host folder containing `build/LichtFeld-Studio` |
| `lfs_repo_container` | `auto` | Container path to mount it at. `auto` probes the image for the user it was built for. Must match the binary's RUNPATH |
| `gpus` | `all` | Passed to `docker --gpus`. Use `device=0` to pin one GPU |
| `extra_run_args` | `[]` | Inserted into every `docker run`, e.g. `["--shm-size=8g"]` |

## `frames`

| Key | Default | Notes |
|---|---|---|
| `enabled` | `true` | Skip when the frames already exist |
| `sample_rate_fps` | `15` | Frames sampled per second of video |
| `rotation` | `0` | `0`, `90` or `-90`, for sideways captures |
| `jpeg_quality` | `100` | Keep high; SfM is sensitive to compression artefacts |
| `resolution` | `original` | `original`, `half`, `quarter`, `eighth`, `custom` |
| `custom_width` | `1280` | Used when `resolution: custom` |
| `trim_start` / `trim_end` | `0.0` | Seconds to drop, e.g. to cut the walk-up to the subject |

### `frames.filter`

Frames are scored by Laplacian variance and the sharpest kept. Selection is spread across the
sequence rather than taken globally, so a well-lit stretch of the orbit cannot starve the
rest of it.

| Key | Default | Notes |
|---|---|---|
| `mode` | `balanced` | `balanced` (even coverage), `quality` (globally sharpest), `custom` |
| `target` | `20%` | A percentage (`20%`) or an absolute count (`300`) |
| `group_size` | `10` | `balanced` only: how many consecutive frames compete; the sharpest of each group survive |
| `groups` | `20` | `custom` only: explicit group count |

`group_size` is the only knob that shapes a balanced selection, and it is worth one
sentence: at the default 20% target, groups of 10 mean *the sharpest 2 of every 10 frames
are kept*. Match it to the keep rate — roughly `2 / keep_rate` — and the survivors stay
evenly spread. Much larger and the two winners of a group can sit side by side, leaving a
hole where the next group's losers were; much smaller and the sharpness test has nothing to
choose between. The GUI computes this live under the control, using the real length of the
clips you loaded, and `gs-recon plan` prints it in the filter step's note.

It replaces 1.0.0's `scalar` (a power-of-two divisor on the group *count*). Configs written
before 1.1.0 are migrated on load — `scalar: 2` at `target: 20%` becomes `group_size: 10`.

## `sfm`

| Key | Default | Notes |
|---|---|---|
| `enabled` | `true` | |
| `camera_model` | `OPENCV` | Models real lens distortion. `PINHOLE` only for already-rectified input |
| `matcher` | `sequential-loop` | Sequential matching plus vocabulary-tree loop closure — right for an orbit that returns to its start |
| `mapper` | `glomap` | GLOMAP is a global solver, much faster than COLMAP's incremental mapper |
| `convert` | `TXT+PLY` | Extra formats written next to the binary model |
| `undistort` | `false` | GUT mode handles distortion natively, so this usually only loses detail |
| `reorganize` | `false` | Promotes `dense/` output into `images/` and `sparse/`. Requires `undistort` |
| `orient` | `false` | Rotates the sparse model so its dominant plane is axis-aligned |
| `loop_overlap` | `30` | Temporal neighbours matched per frame |
| `loop_detection_num_images` | `100` | Loop candidates retrieved from the vocabulary tree |
| `loop_detection_period` | `5` | Check for loop closure every N frames |

The last three apply only to `sequential-loop`. The defaults are tuned for elongated subjects
captured in several passes: a wider window and more candidates let the second pass anchor to
the first where features are sparse.

## `splat`

| Key | Default | Notes |
|---|---|---|
| `enabled` | `true` | |
| `iterations` | `30000` | Diminishing returns past roughly 30k |
| `headless` | `true` | Required over SSH or unattended |
| `max_cap` | `1000000` | Upper bound on Gaussians. `null` lets LichtFeld decide |
| `ppisp` | `true` | Per-camera exposure and white balance. Recommended for orbits |
| `enable_mip` | `true` | Anti-aliasing for mixed near/far distances |
| `bilateral_grid` | `false` | Alternative appearance correction — conflicts with `ppisp`, pick one |
| `undistort` | `false` | Only if COLMAP did not undistort and you are not relying on GUT |
| `extra_args` | `""` | Appended verbatim, e.g. `--strategy mcmc --sh-degree 2` |
| `export_ply` | `auto` | Filename inside the project's `gs/` folder. `auto` resolves to `splat_<iterations>.ply`, matching what LichtFeld writes; a fixed name desynchronises as soon as `iterations` changes |

Training always runs in GUT mode. The release pins one LichtFeld version deliberately: the
old build's flags (`--pose-opt`) no longer exist upstream, and offering a version switch
meant every user had to know which image they had.

### `splat.prune`

Trained splats carry near-transparent Gaussians and an outlier halo seeded by SfM noise.
Both cost memory in every viewer and neither contributes to the render.

| Key | Default | Notes |
|---|---|---|
| `enabled` | `true` | |
| `alpha` | `0.005` | Minimum opacity to keep |
| `percentile` | `99.0` | Radius percentile defining the keep sphere |
| `margin` | `1.0` | Multiplier on that radius |
