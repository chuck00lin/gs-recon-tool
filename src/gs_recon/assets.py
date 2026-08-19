"""Fetch the large binary assets that are too big to ship in the repo."""

from __future__ import annotations

import pathlib
import shutil
import sys
import urllib.error
import urllib.request
from typing import Callable, Optional

from . import env

ProgressFn = Callable[[int, int], None]


def _default_progress(done: int, total: int) -> None:
    if total <= 0:
        sys.stderr.write(f"\r  {done / 1e6:.1f} MB")
    else:
        pct = done / total * 100
        bar_len = 30
        filled = int(bar_len * done / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stderr.write(f"\r  [{bar}] {pct:5.1f}%  {done / 1e6:.1f}/{total / 1e6:.1f} MB")
    sys.stderr.flush()


def vocab_tree_present() -> bool:
    path = env.vocab_tree_path()
    # Guard against a half-finished download from an interrupted setup run.
    return path.is_file() and path.stat().st_size == env.VOCAB_TREE_SIZE


def download_vocab_tree(
    *,
    force: bool = False,
    progress: Optional[ProgressFn] = _default_progress,
) -> pathlib.Path:
    """Download COLMAP's vocabulary tree, used for loop-closure matching."""
    target = env.vocab_tree_path()
    if vocab_tree_present() and not force:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")

    try:
        with urllib.request.urlopen(env.VOCAB_TREE_URL, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download the vocabulary tree from {env.VOCAB_TREE_URL}: {exc}\n"
            f"You can fetch it manually and save it as {target}"
        ) from exc
    finally:
        if progress:
            sys.stderr.write("\n")

    size = tmp.stat().st_size
    if size != env.VOCAB_TREE_SIZE:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded vocabulary tree is {size} bytes, expected "
            f"{env.VOCAB_TREE_SIZE}. The download was truncated or the upstream "
            f"file changed; re-run `gs-recon setup --assets --force`."
        )

    shutil.move(str(tmp), str(target))
    return target


def pull_image(image: str, *, log: Callable[[str], None] = print) -> bool:
    import subprocess

    exe = env.docker_binary()
    if not exe:
        log("docker is not installed -- cannot pull images")
        return False
    log(f"Pulling {image} ...")
    try:
        proc = subprocess.run([exe, "pull", image])
    except OSError as exc:
        log(f"docker pull failed: {exc}")
        return False
    if proc.returncode != 0:
        log(f"docker pull {image} failed with exit code {proc.returncode}")
        return False
    env.docker_image_exists.cache_clear()
    return True
