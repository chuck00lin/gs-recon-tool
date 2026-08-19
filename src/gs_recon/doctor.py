"""Environment self-check.

The goal is that a new user runs `gs-recon doctor` once and is told, in exact
commands, everything they still need to do. Every failing check carries a fix.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from . import assets, env
from .config import Config

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

SYMBOLS = {OK: "✓", WARN: "!", FAIL: "✗", SKIP: "-"}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""

    @property
    def symbol(self) -> str:
        return SYMBOLS.get(self.status, "?")


def run_checks(cfg: Optional[Config] = None) -> list[Check]:
    cfg = cfg or Config()
    checks: list[Check] = []
    checks += _python_checks()
    checks += _docker_checks(cfg)
    checks += _colmap_checks(cfg)
    checks += _lichtfeld_checks(cfg)
    return checks


# ---------------------------------------------------------------------------
def _python_checks() -> list[Check]:
    checks = [Check("Python", OK, f"{sys.version.split()[0]} at {sys.executable}")]

    for module, package, required in (
        ("cv2", "opencv-python", True),
        ("numpy", "numpy", True),
        ("yaml", "PyYAML", True),
        ("tqdm", "tqdm", True),
        ("PyQt6", "PyQt6", False),
    ):
        present = importlib.util.find_spec(module) is not None
        if present:
            checks.append(Check(f"python: {package}", OK))
        elif required:
            checks.append(Check(
                f"python: {package}", FAIL, "not importable",
                f"pip install {package}",
            ))
        else:
            checks.append(Check(
                f"python: {package}", WARN, "not installed -- CLI works, GUI will not start",
                "pip install 'gs-recon-tool[gui]'",
            ))
    return checks


def _docker_checks(cfg: Config) -> list[Check]:
    if not env.docker_binary():
        return [Check(
            "docker", FAIL, "not found on PATH",
            "Install Docker Engine: https://docs.docker.com/engine/install/\n"
            "        then add yourself to the docker group:\n"
            "          sudo usermod -aG docker $USER   # log out and back in",
        )]

    version = env.docker_version()
    if version is None:
        return [Check(
            "docker", FAIL, "installed but the daemon is unreachable",
            "sudo systemctl start docker\n"
            "        (permission denied? sudo usermod -aG docker $USER, then re-login)",
        )]

    checks = [Check("docker", OK, f"server {version}")]

    gpus = env.gpu_names()
    if gpus:
        checks.append(Check("nvidia-smi", OK, "; ".join(gpus)))
    elif shutil.which("nvidia-smi"):
        checks.append(Check("nvidia-smi", WARN, "present but reported no GPUs"))
    else:
        checks.append(Check(
            "nvidia-smi", FAIL, "no NVIDIA driver found on the host",
            "This pipeline needs a CUDA GPU. Install the NVIDIA driver first.",
        ))

    works, detail = env.nvidia_runtime_ok((cfg.docker.colmap_image, cfg.docker.lfs_image))
    checks.append(Check(
        "docker GPU access", OK if works else FAIL, detail,
        "" if works else
        "Install the NVIDIA Container Toolkit:\n"
        "        https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html\n"
        "        then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker",
    ))
    return checks


def _colmap_checks(cfg: Config) -> list[Check]:
    checks: list[Check] = []
    image = cfg.docker.colmap_image
    if env.docker_image_exists(image):
        checks.append(Check(f"image: {image}", OK))
    else:
        checks.append(Check(
            f"image: {image}", FAIL, "not present locally",
            f"gs-recon setup --pull    (or: docker pull {image})",
        ))

    needs_vocab = cfg.sfm.matcher in {"sequential-loop", "vocab_tree"}
    path = env.vocab_tree_path()
    if assets.vocab_tree_present():
        checks.append(Check("vocabulary tree", OK, str(path)))
    elif path.is_file():
        checks.append(Check(
            "vocabulary tree", FAIL,
            f"{path} is {path.stat().st_size} bytes, expected {env.VOCAB_TREE_SIZE} (truncated download)",
            "gs-recon setup --assets --force",
        ))
    else:
        checks.append(Check(
            "vocabulary tree", FAIL if needs_vocab else WARN,
            f"missing: {path}"
            + ("" if needs_vocab else f" (not needed by matcher '{cfg.sfm.matcher}')"),
            "gs-recon setup --assets",
        ))
    return checks


def _lichtfeld_checks(cfg: Config) -> list[Check]:
    if not cfg.splat.enabled:
        return [Check("LichtFeld Studio", SKIP, "splat stage disabled in config")]

    checks: list[Check] = []
    image = cfg.docker.lfs_image
    image_ok = env.docker_image_exists(image)
    if image_ok:
        checks.append(Check(f"image: {image}", OK))
    else:
        checks.append(Check(
            f"image: {image}", FAIL,
            "not present locally (this image is built per machine, not pulled)",
            "git clone https://github.com/MrNeRF/LichtFeld-Studio\n"
            "        cd LichtFeld-Studio && ./docker/run_docker.sh -bu",
        ))

    repo = cfg.docker.resolved_lfs_repo_host()
    if repo is None:
        checks.append(Check(
            "LichtFeld checkout", FAIL, "not found in any of the usual places",
            "Set docker.lfs_repo_host in your config, or export\n"
            "        LICHTFELD_STUDIO_ROOT=/path/to/LichtFeld-Studio",
        ))
        return checks

    binary = repo / "build" / "LichtFeld-Studio"
    if binary.is_file():
        checks.append(Check("LichtFeld checkout", OK, str(repo)))
    else:
        checks.append(Check(
            "LichtFeld checkout", FAIL,
            f"found {repo} but there is no build/LichtFeld-Studio",
            f"cd {repo} && ./docker/run_docker.sh -bu\n"
            f"        then build the project inside the container",
        ))
        return checks

    container_path = cfg.docker.resolved_lfs_repo_container()
    expected = _runpath_project_dir(binary)
    if expected and expected != container_path:
        checks.append(Check(
            "LichtFeld mount path", FAIL,
            f"binary RUNPATH expects {expected}, config resolves to {container_path}",
            f"Set docker.lfs_repo_container: {expected}",
        ))
    else:
        checks.append(Check("LichtFeld mount path", OK, container_path))
    return checks


def _runpath_project_dir(binary: pathlib.Path) -> Optional[str]:
    """Read the container-side project path baked into the binary's RUNPATH.

    This is the single most confusing failure mode -- mount the checkout at the
    wrong path and it dies with 'cannot open shared object file' -- so we read
    the ground truth out of the ELF instead of guessing.
    """
    if not shutil.which("objdump"):
        return None
    try:
        out = subprocess.run(
            ["objdump", "-x", str(binary)],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("RUNPATH", "RPATH")):
            continue
        for entry in stripped.split(None, 1)[-1].split(":"):
            marker = "/build/vcpkg_installed/"
            if marker in entry:
                return entry.split(marker)[0]
    return None


# ---------------------------------------------------------------------------
def format_report(checks: list[Check], *, color: bool = True) -> str:
    def paint(text: str, status: str) -> str:
        if not color:
            return text
        codes = {OK: "32", WARN: "33", FAIL: "31", SKIP: "90"}
        return f"\033[{codes.get(status, '0')}m{text}\033[0m"

    width = max((len(c.name) for c in checks), default=0)
    lines: list[str] = []
    for check in checks:
        head = f"  {paint(check.symbol, check.status)} {check.name.ljust(width)}"
        lines.append(f"{head}  {check.detail}".rstrip())
        if check.fix and check.status in (FAIL, WARN):
            for fix_line in check.fix.splitlines():
                lines.append(f"      -> {fix_line}" if fix_line == check.fix.splitlines()[0]
                             else f"      {fix_line}")

    failures = sum(1 for c in checks if c.status == FAIL)
    warnings = sum(1 for c in checks if c.status == WARN)
    lines.append("")
    if failures:
        lines.append(paint(f"{failures} blocking problem(s), {warnings} warning(s).", FAIL))
        lines.append("Fix the items marked -> above, then re-run `gs-recon doctor`.")
    elif warnings:
        lines.append(paint(f"Ready to run, with {warnings} warning(s).", WARN))
    else:
        lines.append(paint("All checks passed. You are ready to run.", OK))
    return "\n".join(lines)


def has_blocking_failures(checks: list[Check]) -> bool:
    return any(c.status == FAIL for c in checks)
