# gs-recon-tool

Turn a video of an object into a 3D Gaussian Splatting model, in one command or one window.

```
video ──▶ frames ──▶ COLMAP/GLOMAP SfM ──▶ LichtFeld Studio 3DGS ──▶ pruned .ply
```

Every heavy tool runs in Docker, so the only thing you install on your machine is a small
Python package. There is a CLI for batch and headless work and a GUI for everything else —
both drive the exact same code, so a run designed in the GUI can be saved and replayed by
the CLI without change.

---

## 快速開始（實驗室成員）

```bash
# 1. 安裝（pipx 會自動建立隔離環境，不會汙染你的 conda）
pipx install "gs-recon-tool[gui] @ git+https://github.com/chuck00lin/gs-recon-tool"

# 2. 檢查這台機器缺什麼 —— 每個問題都會附上修復指令
gs-recon doctor

# 3. 一次補齊（下載 COLMAP 映像 + 詞彙樹，約 14 GB + 118 MB）
gs-recon setup --all

# 4. 開圖形介面
gs-recon gui
```

只有 3DGS 訓練那一段需要額外準備：`lichtfeld-studio` 映像**不能 pull**，要在自己機器上建，
見 [docs/INSTALL.md](docs/INSTALL.md#lichtfeld-studio)。前兩段（影格擷取、SfM）不需要它就能跑。

---

## Install

Requires Linux, an NVIDIA GPU, Docker with the NVIDIA Container Toolkit, and Python 3.9+.

```bash
pipx install "gs-recon-tool[gui] @ git+https://github.com/chuck00lin/gs-recon-tool"
gs-recon doctor          # tells you exactly what is still missing
gs-recon setup --all     # pulls images and downloads the vocabulary tree
```

Full details, including how to build the LichtFeld Studio image, are in
[docs/INSTALL.md](docs/INSTALL.md).

## Use it

### GUI

```bash
gs-recon gui
```

- **Left** — what you are reconstructing: add videos or existing project folders, and watch
  each step's status as it runs.
- **Right** — how: one tab per stage, plus an *Advanced* tab holding the Docker settings you
  should never need to open.
- **Bottom** — what is happening now: progress, live log, and Start / Stop / Resume.

Useful buttons: **Check environment** runs the same checks as `gs-recon doctor`;
**Preview full plan** shows every command before anything executes; **Save config** writes a
YAML the CLI can replay; **Copy CLI command** hands you the equivalent one-liner.

### CLI

```bash
gs-recon plan capture.mp4                    # print the commands, run nothing
gs-recon run capture.mp4                     # the whole pipeline
gs-recon run /data/day01 --recursive         # every video in every subfolder
gs-recon run ./plantA-frames --only splat    # re-train an existing reconstruction
gs-recon run capture.mp4 --set splat.iterations=15000
gs-recon run capture.mp4 --start-at 4        # resume after fixing a failure
```

Anything a video produces lands next to it:

```
capture.mp4
capture-frames/
├── images/            # extracted and sharpness-filtered frames
├── database.db        # COLMAP features and matches
├── sparse/0/          # camera poses + sparse cloud (.bin, .txt, points.ply)
└── gs/
    ├── checkpoints/
    └── splat_30000.ply
```

## Configure

Every setting lives in one YAML file. Generate a starting point, then edit it — or design it
in the GUI and press *Save config*.

```bash
gs-recon init                       # writes ~/.config/gs-recon/config.yaml
gs-recon run capture.mp4 -c my.yaml
```

See [docs/CONFIG.md](docs/CONFIG.md) for every key, and
[examples/example-config.yaml](examples/example-config.yaml) for an annotated file.

## Presets

The GUI ships three starting points, chosen from the toolbar:

| Preset | Sample rate | Frames kept | Iterations | Roughly |
|---|---|---|---|---|
| Fast draft | 10 fps | 15% | 7 000 | check the capture worked |
| Balanced *(default)* | 15 fps | 20% | 30 000 | everyday reconstruction |
| High quality | 20 fps | 35% | 50 000 | final figures |

## When something breaks

The pipeline stops at the failing step and tells you how to resume from it — you never redo
the hours that already succeeded. `gs-recon doctor` diagnoses the environment;
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) covers the failures that are not the
environment's fault.

## What this replaces

This is a packaged, portable version of `pipeline_assist_gui.py` from the internal
`3dplant-workflow` repository. The behaviour is the same; what changed is that hard-coded
paths became configuration, the command builders were separated from the Qt widgets so a CLI
became possible, and the LichtFeld container is now started per run with your dataset
explicitly bind-mounted — which is what lets it work on a machine that is not the one it was
written on.

## Licence

MIT — see [LICENSE](LICENSE).
