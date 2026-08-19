"""Host environment probing.

Everything in this module is read-only and side-effect free apart from a small
in-process cache. It answers the questions the rest of the tool needs before it
can build a working ``docker run`` command on a machine it has never seen.
"""

from __future__ import annotations

import functools
import os
import pathlib
import re
import shutil
import subprocess
from typing import Optional

# Where `gs-recon setup --assets` puts downloaded files, and where the GUI/CLI
# look for them. Overridable so a shared lab install can point everyone at one
# copy of the 118 MB vocabulary tree instead of N copies.
DEFAULT_DATA_HOME = pathlib.Path(
    os.environ.get("GS_RECON_HOME")
    or (pathlib.Path(os.environ.get("XDG_DATA_HOME", pathlib.Path.home() / ".local/share")) / "gs-recon")
)

CONFIG_HOME = pathlib.Path(
    os.environ.get("GS_RECON_CONFIG_HOME")
    or (pathlib.Path(os.environ.get("XDG_CONFIG_HOME", pathlib.Path.home() / ".config")) / "gs-recon")
)

VOCAB_TREE_NAME = "vocab_tree_flickr100K_words256K.bin"
VOCAB_TREE_URL = f"https://demuc.de/colmap/{VOCAB_TREE_NAME}"
VOCAB_TREE_SIZE = 117893693  # bytes, used as a cheap integrity check


def vocab_tree_path() -> pathlib.Path:
    return DEFAULT_DATA_HOME / "vocab_trees" / VOCAB_TREE_NAME


def user_config_path() -> pathlib.Path:
    return CONFIG_HOME / "config.yaml"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def uid() -> int:
    return os.getuid()


def gid() -> int:
    return os.getgid()


def username() -> str:
    return os.environ.get("USER") or os.environ.get("LOGNAME") or pathlib.Path.home().name


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
def docker_binary() -> Optional[str]:
    return shutil.which("docker")


@functools.lru_cache(maxsize=1)
def docker_version() -> Optional[str]:
    exe = docker_binary()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    version = out.stdout.strip()
    return version or None


@functools.lru_cache(maxsize=None)
def docker_image_exists(image: str) -> bool:
    exe = docker_binary()
    if not exe:
        return False
    try:
        out = subprocess.run(
            [exe, "image", "inspect", image],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


@functools.lru_cache(maxsize=None)
def nvidia_runtime_ok(candidates: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Return (works, detail) for the NVIDIA container runtime.

    Tries the smallest locally-available image first so a fresh machine is not
    made to pull several GB just to answer "can containers see the GPU?".
    """
    exe = docker_binary()
    if not exe:
        return False, "docker not installed"
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi not found on host (no NVIDIA driver?)"

    images = ("ubuntu:22.04", "nvidia/cuda:12.4.0-base-ubuntu22.04", *candidates)
    for image in images:
        if not docker_image_exists(image):
            continue
        try:
            out = subprocess.run(
                [exe, "run", "--rm", "--gpus", "all", image, "nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        if out.returncode == 0:
            gpus = [ln.strip() for ln in out.stdout.splitlines() if ln.strip().startswith("GPU")]
            summary = gpus[0] if gpus else "GPU visible"
            return True, f"{summary}  (verified with {image})"
        return False, (out.stderr or out.stdout).strip()[:300]
    return False, "no local image available to test with -- run `gs-recon setup --pull` first"


@functools.lru_cache(maxsize=None)
def image_home_username(image: str) -> Optional[str]:
    """Ask an image which non-root user it was built for.

    The LichtFeld-Studio image bakes the builder's username into the image (and
    the binary's RUNPATH), so the container-side project path differs per
    machine. Probing beats guessing.
    """
    exe = docker_binary()
    if not exe or not docker_image_exists(image):
        return None
    try:
        out = subprocess.run(
            [exe, "run", "--rm", "--entrypoint", "/bin/bash", image, "-lc", "id -un"],
            capture_output=True, text=True, timeout=90,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    name = out.stdout.strip().splitlines()[-1].strip() if out.stdout.strip() else ""
    return name or None


def gpu_names() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Host memory (surfaced in the GUI so users can size --max-cap sensibly)
# ---------------------------------------------------------------------------
def host_memory() -> dict[str, float]:
    info: dict[str, float] = {}
    meminfo = pathlib.Path("/proc/meminfo")
    if not meminfo.exists():
        return info
    pattern = re.compile(r"^(MemTotal|MemFree|MemAvailable):\s+(\d+)\s+kB$")
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line.strip())
            if match:
                key, value_kb = match.groups()
                info[f"mem_{key[3:].lower()}_kb"] = float(value_kb)
    except OSError:
        return info
    for key in list(info):
        info[key.replace("_kb", "_gb")] = info[key] / (1024.0 * 1024.0)
    return info


# ---------------------------------------------------------------------------
# LichtFeld-Studio checkout discovery
# ---------------------------------------------------------------------------
LFS_SEARCH_DIRS = ("github", "projects", "src", "repos", "code", "work")


def find_lichtfeld_repo() -> Optional[pathlib.Path]:
    """Best-effort autodetect of a *built* LichtFeld-Studio checkout."""
    env_hint = os.environ.get("LICHTFELD_STUDIO_ROOT")
    candidates: list[pathlib.Path] = []
    if env_hint:
        candidates.append(pathlib.Path(env_hint).expanduser())
    home = pathlib.Path.home()
    for parent in LFS_SEARCH_DIRS:
        candidates.append(home / parent / "LichtFeld-Studio")
    candidates.append(home / "LichtFeld-Studio")
    for candidate in candidates:
        if (candidate / "build" / "LichtFeld-Studio").is_file():
            return candidate.resolve()
    # Fall back to an unbuilt checkout so `doctor` can give a precise diagnosis
    # ("found the source but no build/") instead of "not found anywhere".
    for candidate in candidates:
        if (candidate / "docker" / "run_docker.sh").is_file():
            return candidate.resolve()
    return None
