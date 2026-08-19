"""Configuration model shared by the CLI and the GUI.

One schema, one file format. The GUI is a config editor with a Run button; the
CLI reads the same YAML. Anything the GUI can do, a saved config can reproduce
headlessly -- that is what makes "design it on the laptop, run it overnight on
the workstation" work.
"""

from __future__ import annotations

import copy
import dataclasses
import pathlib
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional

import yaml

from . import env

CONFIG_VERSION = 1

RESOLUTIONS = ("original", "half", "quarter", "eighth", "custom")
FILTER_MODES = ("balanced", "quality", "custom")
CAMERA_MODELS = (
    "OPENCV", "PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL",
    "RADIAL", "SIMPLE_RADIAL_FISHEYE",
)
MATCHERS = ("sequential-loop", "sequential", "exhaustive", "vocab_tree", "spatial")
MAPPERS = ("glomap", "colmap")
CONVERT_FORMATS = ("TXT+PLY", "PLY", "TXT", "NVM", "Bundler", "VRML", "X3D")


# ---------------------------------------------------------------------------
@dataclass
class DockerConfig:
    """How to reach the two containerised toolchains.

    ``lfs_repo_host`` matters more than it looks: the LichtFeld-Studio image
    ships *no* binary. The executable lives in the host checkout's ``build/``
    and its RUNPATH is baked to ``/home/<builder>/projects/LichtFeld-Studio``,
    so the checkout has to be bind-mounted at exactly that container path.
    ``lfs_repo_container: auto`` derives it by probing the image.
    """

    colmap_image: str = "jinwj1996/glomap"
    lfs_image: str = "lichtfeld-studio:latest"
    lfs_repo_host: str = ""
    lfs_repo_container: str = "auto"
    gpus: str = "all"          # "all", "device=0", "device=0,1", or "" to disable
    extra_run_args: list[str] = field(default_factory=list)

    def resolved_lfs_repo_host(self) -> Optional[pathlib.Path]:
        if self.lfs_repo_host:
            return pathlib.Path(self.lfs_repo_host).expanduser()
        return env.find_lichtfeld_repo()

    def resolved_lfs_repo_container(self) -> str:
        if self.lfs_repo_container and self.lfs_repo_container != "auto":
            return self.lfs_repo_container
        # Probe the image for the user it was built for; fall back to the host
        # username, which is right whenever the image was built on this machine.
        name = env.image_home_username(self.lfs_image) or env.username()
        return f"/home/{name}/projects/LichtFeld-Studio"


@dataclass
class FilterConfig:
    mode: str = "balanced"
    target: str = "20%"        # "20%" (percentage) or "300" (absolute count)
    scalar: int = 2            # used by mode=balanced
    groups: int = 20           # used by mode=custom

    def target_is_percentage(self) -> bool:
        return str(self.target).strip().endswith("%")

    def target_value(self) -> float:
        return float(str(self.target).strip().rstrip("%"))


@dataclass
class FramesConfig:
    enabled: bool = True
    sample_rate_fps: int = 15
    rotation: int = 0          # 0, 90, -90
    jpeg_quality: int = 100
    resolution: str = "original"
    custom_width: int = 1280
    trim_start: float = 0.0
    trim_end: float = 0.0
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class SfmConfig:
    enabled: bool = True
    camera_model: str = "OPENCV"
    matcher: str = "sequential-loop"
    mapper: str = "glomap"
    # GUT mode handles lens distortion natively, so undistortion is off by
    # default. Turn both on together if you ever need an undistorted dataset.
    undistort: bool = False
    reorganize: bool = False
    orient: bool = False
    convert: str = "TXT+PLY"
    # sequential-loop tuning (elongated subjects with multi-loop captures)
    loop_overlap: int = 30
    loop_detection_num_images: int = 100
    loop_detection_period: int = 5


@dataclass
class PruneConfig:
    enabled: bool = True
    alpha: float = 0.005
    percentile: float = 99.0
    margin: float = 1.0


@dataclass
class SplatConfig:
    enabled: bool = True
    iterations: int = 30000
    headless: bool = True
    max_cap: Optional[int] = 1000000   # None -> let LichtFeld decide
    ppisp: bool = True
    enable_mip: bool = True
    bilateral_grid: bool = False
    undistort: bool = False
    extra_args: str = ""
    # "auto" tracks the iteration count, which is what LichtFeld names its own
    # output. A fixed name here silently desynchronises the moment someone
    # changes `iterations`, leaving the prune step pointing at nothing.
    export_ply: str = "auto"
    prune: PruneConfig = field(default_factory=PruneConfig)

    def export_ply_name(self) -> str:
        name = (self.export_ply or "auto").strip()
        if name in ("", "auto"):
            return f"splat_{self.iterations}.ply"
        return name


@dataclass
class Config:
    version: int = CONFIG_VERSION
    docker: DockerConfig = field(default_factory=DockerConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    sfm: SfmConfig = field(default_factory=SfmConfig)
    splat: SplatConfig = field(default_factory=SplatConfig)

    # ---------------------------------------------------------------- I/O --
    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    def dump_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save(self, path: pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dump_yaml(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _from_dict(cls, data or {})

    @classmethod
    def load(cls, path: pathlib.Path) -> "Config":
        raw = yaml.safe_load(pathlib.Path(path).expanduser().read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        version = raw.get("version", CONFIG_VERSION)
        if version != CONFIG_VERSION:
            raise ValueError(
                f"{path}: config version {version} is not supported by this "
                f"release (expected {CONFIG_VERSION}). Run `gs-recon init` to "
                f"generate a fresh config."
            )
        return cls.from_dict(raw)

    @classmethod
    def resolve(cls, explicit: Optional[pathlib.Path] = None) -> "Config":
        """Load an explicit config, else the per-user one, else defaults."""
        if explicit is not None:
            return cls.load(explicit)
        user = env.user_config_path()
        if user.is_file():
            return cls.load(user)
        return cls()

    # ------------------------------------------------------------ helpers --
    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty means fine)."""
        problems: list[str] = []
        if self.frames.resolution not in RESOLUTIONS:
            problems.append(f"frames.resolution must be one of {RESOLUTIONS}")
        if self.frames.rotation not in (0, 90, -90):
            problems.append("frames.rotation must be 0, 90 or -90")
        if not 1 <= self.frames.jpeg_quality <= 100:
            problems.append("frames.jpeg_quality must be between 1 and 100")
        if self.frames.filter.mode not in FILTER_MODES:
            problems.append(f"frames.filter.mode must be one of {FILTER_MODES}")
        try:
            value = self.frames.filter.target_value()
            if value <= 0:
                problems.append("frames.filter.target must be positive")
            if self.frames.filter.target_is_percentage() and value > 100:
                problems.append("frames.filter.target percentage must be <= 100%")
        except ValueError:
            problems.append(
                f"frames.filter.target {self.frames.filter.target!r} is not a "
                f"number or percentage (e.g. '20%' or '300')"
            )
        if self.sfm.camera_model not in CAMERA_MODELS:
            problems.append(f"sfm.camera_model must be one of {CAMERA_MODELS}")
        if self.sfm.matcher not in MATCHERS:
            problems.append(f"sfm.matcher must be one of {MATCHERS}")
        if self.sfm.mapper not in MAPPERS:
            problems.append(f"sfm.mapper must be one of {MAPPERS}")
        if self.sfm.convert not in CONVERT_FORMATS:
            problems.append(f"sfm.convert must be one of {CONVERT_FORMATS}")
        if self.sfm.reorganize and not self.sfm.undistort:
            problems.append(
                "sfm.reorganize moves the output of sfm.undistort, so it needs "
                "sfm.undistort enabled (or an earlier undistort run)"
            )
        if self.splat.iterations < 1:
            problems.append("splat.iterations must be >= 1")
        if self.splat.ppisp and self.splat.bilateral_grid:
            problems.append(
                "splat.ppisp and splat.bilateral_grid both correct appearance; "
                "enable one, not both"
            )
        if not (self.frames.enabled or self.sfm.enabled or self.splat.enabled):
            problems.append("all three stages are disabled -- nothing to run")
        return problems


# ---------------------------------------------------------------------------
# Minimal dataclass <-> dict codec (keeps the YAML clean and comment-free)
# ---------------------------------------------------------------------------
def _as_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"unknown key(s) in {cls.__name__.replace('Config', '').lower() or 'config'}: "
            + ", ".join(sorted(unknown))
        )
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        ftype = f.type
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _from_dict(ftype, value)
        elif dataclasses.is_dataclass(f.default_factory() if f.default_factory is not dataclasses.MISSING else None) and isinstance(value, dict):
            kwargs[name] = _from_dict(type(f.default_factory()), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
