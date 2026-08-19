# Installing gs-recon-tool

## What you need first

| Requirement | Why | Check |
|---|---|---|
| Linux | The Docker images and the LichtFeld binary are Linux-only | `uname -s` |
| NVIDIA GPU | Both SfM and 3DGS training are CUDA-only | `nvidia-smi` |
| Docker Engine | Every heavy tool runs in a container | `docker version` |
| NVIDIA Container Toolkit | Lets containers see the GPU | `docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L` |
| Python 3.9+ | The tool itself | `python3 --version` |

Disk: about 15 GB for the COLMAP/GLOMAP image, another ~11 GB if you build the LichtFeld
image, plus your data.

### Docker without sudo

```bash
sudo usermod -aG docker $USER
# log out and back in, then confirm:
docker run --rm hello-world
```

### NVIDIA Container Toolkit

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L   # should list your GPU
```

## Install the tool

This repository is private. Before installing, make sure your GitHub account has been added
as a collaborator and that this machine's SSH key is registered with GitHub:

```bash
ssh -T git@github.com     # should greet you by username
```

If it does not, generate a key and add the public half at <https://github.com/settings/keys>:

```bash
ssh-keygen -t ed25519 -C "$USER@$(hostname)"
cat ~/.ssh/id_ed25519.pub
```

`pipx` is recommended for the install itself: it puts the tool in its own virtual environment,
so it cannot break (or be broken by) your conda environments.

```bash
python3 -m pip install --user pipx && python3 -m pipx ensurepath
pipx install "gs-recon-tool[gui] @ git+ssh://git@github.com/chuck00lin/gs-recon-tool"
```

<details>
<summary>Alternatives</summary>

```bash
# plain pip, into whatever environment is active
pip install "gs-recon-tool[gui] @ git+ssh://git@github.com/chuck00lin/gs-recon-tool"

# from a local clone, editable, for development
git clone git@github.com:chuck00lin/gs-recon-tool.git
cd gs-recon-tool && pip install -e ".[gui,dev]"

# upgrading later
pipx upgrade gs-recon-tool
```

Drop `[gui]` if you only want the CLI — that skips the PyQt6 dependency.
</details>

## Fetch what the pipeline needs

```bash
gs-recon doctor        # what is missing, and the command to fix each item
gs-recon setup --all   # pull the COLMAP image, download the vocabulary tree
```

`setup --all` does two things:

- **`jinwj1996/glomap`** (~14 GB) — COLMAP and GLOMAP. Public, pulled from Docker Hub.
- **The vocabulary tree** (118 MB) — needed by the `sequential-loop` and `vocab_tree`
  matchers for loop closure. Downloaded once to `~/.local/share/gs-recon/vocab_trees/`.

If a colleague already has the vocabulary tree, copy theirs instead of downloading again:

```bash
gs-recon setup --assets --from /path/to/vocab_tree_flickr100K_words256K.bin
```

Set `GS_RECON_HOME` to point several users at one shared copy.

## LichtFeld Studio

**This image cannot be pulled.** It is built on your machine, with your username and UID
baked in, and it contains no LichtFeld binary — the executable lives in the checkout's
`build/` directory and is produced by building inside the container.

```bash
git clone https://github.com/MrNeRF/LichtFeld-Studio
cd LichtFeld-Studio
./docker/run_docker.sh -bu        # build the image and enter the container
# inside the container, build the project (see its README), then exit
```

Afterwards `build/LichtFeld-Studio` should exist:

```bash
ls -l build/LichtFeld-Studio
gs-recon doctor                   # should now find the checkout
```

`gs-recon` autodetects a checkout under `~/github`, `~/projects`, `~/src`, `~/repos`,
`~/code`, `~/work` or `~`. Anywhere else, point at it explicitly:

```bash
export LICHTFELD_STUDIO_ROOT=/data/tools/LichtFeld-Studio
# or set docker.lfs_repo_host in your config
```

### Why the mount path matters

The binary's `RUNPATH` contains an absolute path — `/home/<builder>/projects/LichtFeld-Studio/build/vcpkg_installed/x64-linux/lib`
— so the checkout has to be bind-mounted at exactly that path inside the container or it
dies with `error while loading shared libraries`. `gs-recon` derives the path by asking the
image which user it was built for, and `gs-recon doctor` verifies it against the RUNPATH
actually recorded in your binary. If they disagree it tells you the value to set.

**You do not need any of this for the frames and SfM stages.** Uncheck ③ Splat in the GUI,
or run `gs-recon run video.mp4 --only frames,sfm`, and get a COLMAP reconstruction with no
LichtFeld involvement at all.

## Verify

```bash
gs-recon doctor      # everything green
gs-recon gui         # window opens
```
