# Troubleshooting

Start with `gs-recon doctor`. It catches every environment problem below and prints the fix.
This page is for the failures it cannot see.

---

## Setup

### `permission denied while trying to connect to the Docker daemon socket`

You are not in the `docker` group.

```bash
sudo usermod -aG docker $USER   # then log out and back in
```

### `could not select device driver "nvidia"`

The NVIDIA Container Toolkit is missing or not configured.

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### `error while loading shared libraries: libcuda.so.1`

The container was started without GPU access. If you see this from `gs-recon`, check that
`docker.gpus` is not empty in your config.

### `error while loading shared libraries` naming a vcpkg path

The LichtFeld checkout is mounted at the wrong container path. The binary's RUNPATH is
absolute, so the mount point is not a free choice. `gs-recon doctor` reads the RUNPATH out of
your binary and prints the exact value to put in `docker.lfs_repo_container`.

### `image: lichtfeld-studio:latest — not present locally`

This image is built per machine and cannot be pulled. See
[INSTALL.md](INSTALL.md#lichtfeld-studio). You can run the frames and SfM stages without it:

```bash
gs-recon run video.mp4 --only frames,sfm
```

---

## Reconstruction quality

### SfM registers only some of the images

Usually too few or too blurry frames.

- Raise `frames.sample_rate_fps` and `frames.filter.target` — more input to match against.
- Keep `matcher: sequential-loop` and raise `sfm.loop_overlap`.
- Check `images/` by eye. If the frames are motion-blurred, the capture needs redoing: move
  the camera more slowly, and keep the subject filling the frame.

### `mapper` produces several models in `sparse/`

The capture broke into disconnected chunks — the camera lost track somewhere. `sparse/0` is
the largest model and is what later stages use. Re-capturing with a continuous orbit is the
real fix; raising `loop_detection_num_images` sometimes bridges the gap.

### The splat is a fog of floaters

- Confirm SfM succeeded first: open `sparse/0/points.ply`. If the sparse cloud is bad, no
  amount of training will rescue it.
- Raise `splat.prune.alpha` (0.01–0.02) to cut more low-opacity Gaussians.
- Lower `splat.prune.percentile` (95–98) to trim the outlier halo more aggressively.

### Colours shift as the camera moves around the subject

Enable `splat.ppisp` (on by default). Do not also enable `bilateral_grid` — they correct the
same thing and split the capacity between them.

---

## Runtime

### Out of GPU memory during training

- Lower `splat.max_cap` (try 500 000).
- Downscale the frames: `frames.resolution: half`.
- Pin one GPU if others are busy: `docker.gpus: device=0`.

### Training is killed with no error

Usually the host running out of RAM rather than VRAM. Check `dmesg | tail` for the OOM
killer. Fewer frames or `frames.resolution: half` both help.

### A run failed halfway

Nothing already completed is repeated. The CLI prints the step number to resume from:

```bash
gs-recon run video.mp4 --start-at 7
```

The GUI shows a **Resume from step N** button.

### Stopping leaves a container running

`gs-recon` terminates the whole process group, so `docker run --rm` containers stop and clean
themselves up. If one survives a hard kill:

```bash
docker ps          # find it
docker stop <id>
```

---

## Still stuck

Include this in your report — it captures the environment and the exact failing command:

```bash
gs-recon doctor --no-color
gs-recon plan <your inputs>
```
