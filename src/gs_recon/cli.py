"""Command-line entry point for gs-recon-tool."""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import signal
import sys
from typing import Optional

from . import __version__, assets, doctor, env
from .config import Config
from .pipeline import build_plan, discover_inputs, ensure_project_dirs
from .runner import Runner
from .stages.splat import SplatConfigError

STAGES = ("frames", "sfm", "splat")


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gs-recon",
        description=(
            "Video -> COLMAP/GLOMAP SfM -> 3D Gaussian Splatting, in one command.\n"
            "Start with `gs-recon doctor` to check your machine."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  gs-recon doctor\n"
            "  gs-recon setup --all\n"
            "  gs-recon plan capture.mp4\n"
            "  gs-recon run capture.mp4\n"
            "  gs-recon run /data/day01 --recursive --only sfm,splat\n"
            "  gs-recon gui\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"gs-recon-tool {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- doctor ---------------------------------------------------------------
    p_doc = sub.add_parser("doctor", help="Check that this machine can run the pipeline")
    _add_config_arg(p_doc)
    p_doc.add_argument("--no-color", action="store_true")

    # -- setup ----------------------------------------------------------------
    p_setup = sub.add_parser("setup", help="Pull Docker images and download assets")
    _add_config_arg(p_setup)
    p_setup.add_argument("--pull", action="store_true", help="Pull the COLMAP/GLOMAP image")
    p_setup.add_argument("--assets", action="store_true", help="Download the vocabulary tree")
    p_setup.add_argument("--all", action="store_true", help="Everything (--pull --assets)")
    p_setup.add_argument("--force", action="store_true", help="Re-download even if present")
    p_setup.add_argument(
        "--from", dest="from_path", metavar="PATH",
        help="Import the vocabulary tree from a local copy instead of downloading",
    )

    # -- init -----------------------------------------------------------------
    p_init = sub.add_parser("init", help="Write a starter config file")
    p_init.add_argument(
        "-o", "--output", type=pathlib.Path,
        help=f"Where to write it (default: {env.user_config_path()})",
    )
    p_init.add_argument("--force", action="store_true", help="Overwrite an existing file")
    p_init.add_argument("--stdout", action="store_true", help="Print instead of writing")

    # -- plan / run -----------------------------------------------------------
    for name, help_text in (
        ("plan", "Show exactly what would run, without running it"),
        ("run", "Execute the pipeline"),
    ):
        p = sub.add_parser(name, help=help_text)
        _add_config_arg(p)
        p.add_argument(
            "inputs", nargs="+",
            help="Video files, a folder of videos, or existing project folders",
        )
        p.add_argument(
            "--only", metavar="STAGES",
            help=f"Comma-separated subset of stages to run ({', '.join(STAGES)})",
        )
        p.add_argument(
            "-r", "--recursive", action="store_true",
            help="Also scan immediate subfolders of each input folder for videos",
        )
        p.add_argument(
            "--set", dest="overrides", action="append", metavar="KEY=VALUE", default=[],
            help="Override one config value, e.g. --set splat.iterations=15000",
        )
        if name == "run":
            p.add_argument(
                "--start-at", type=int, default=1, metavar="N",
                help="Resume from step N of the printed plan (1-based)",
            )
            p.add_argument("--dry-run", action="store_true", help="Print commands, execute nothing")
            p.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation")
            p.add_argument(
                "--skip-doctor", action="store_true",
                help="Do not run environment checks before starting",
            )

    # -- gui ------------------------------------------------------------------
    p_gui = sub.add_parser("gui", help="Launch the graphical interface")
    _add_config_arg(p_gui)

    # -- passthrough tools ----------------------------------------------------
    p_frames = sub.add_parser(
        "frames", help="Frame extraction / filtering tool (standalone)",
        add_help=False,
    )
    p_frames.add_argument("args", nargs=argparse.REMAINDER)

    p_prune = sub.add_parser(
        "prune", help="Prune a Gaussian-splat PLY (standalone)", add_help=False,
    )
    p_prune.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", type=pathlib.Path,
        help=f"Config file (default: {env.user_config_path()} if it exists, else built-in defaults)",
    )


# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {
        "doctor": cmd_doctor,
        "setup": cmd_setup,
        "init": cmd_init,
        "plan": cmd_plan,
        "run": cmd_run,
        "gui": cmd_gui,
        "frames": cmd_frames,
        "prune": cmd_prune,
    }[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (SplatConfigError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
def _load_config(args) -> Config:
    cfg = Config.resolve(getattr(args, "config", None))
    for override in getattr(args, "overrides", []) or []:
        _apply_override(cfg, override)
    problems = cfg.validate()
    if problems:
        raise ValueError(
            "config is not valid:\n  - " + "\n  - ".join(problems)
        )
    return cfg


def _apply_override(cfg: Config, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"--set expects KEY=VALUE, got {override!r}")
    key, _, raw = override.partition("=")
    target = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"--set {key}: no such config section {part!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"--set {key}: no such config key {leaf!r}")

    current = getattr(target, leaf)
    value: object = raw
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"--set {key}: expected a boolean, got {raw!r}")
        value = lowered in {"true", "1", "yes"}
    elif isinstance(current, int) and not isinstance(current, bool):
        value = int(raw)
    elif isinstance(current, float):
        value = float(raw)
    elif isinstance(current, list):
        value = [item for item in raw.split(",") if item]
    setattr(target, leaf, value)


def _parse_only(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    stages = [s.strip() for s in value.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(
            f"--only: unknown stage(s) {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(STAGES)}"
        )
    return stages


def _plan_from_args(args) -> tuple[Config, "object"]:
    cfg = _load_config(args)
    inputs, warnings = discover_inputs(args.inputs, recursive=args.recursive)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not inputs:
        raise ValueError("no usable inputs found")
    plan = build_plan(cfg, inputs, only=_parse_only(args.only))
    return cfg, plan


# ---------------------------------------------------------------------------
def cmd_doctor(args) -> int:
    cfg = Config.resolve(args.config)
    checks = doctor.run_checks(cfg)
    color = sys.stdout.isatty() and not args.no_color
    print(f"gs-recon-tool {__version__} -- environment check\n")
    print(doctor.format_report(checks, color=color))
    return 1 if doctor.has_blocking_failures(checks) else 0


def cmd_setup(args) -> int:
    cfg = Config.resolve(args.config)
    do_pull = args.pull or args.all
    do_assets = args.assets or args.all or bool(args.from_path)
    if not (do_pull or do_assets):
        do_pull = do_assets = True

    failed = False
    if do_pull:
        print(f"== Docker images ==")
        if not assets.pull_image(cfg.docker.colmap_image):
            failed = True
        if not env.docker_image_exists(cfg.docker.lfs_image):
            print(
                f"\nnote: {cfg.docker.lfs_image} cannot be pulled -- it is built per\n"
                f"      machine from the LichtFeld-Studio checkout:\n"
                f"        git clone https://github.com/MrNeRF/LichtFeld-Studio\n"
                f"        cd LichtFeld-Studio && ./docker/run_docker.sh -bu"
            )

    if do_assets:
        print(f"\n== Vocabulary tree ==")
        target = env.vocab_tree_path()
        if args.from_path:
            source = pathlib.Path(args.from_path).expanduser()
            if not source.is_file():
                print(f"error: {source} is not a file", file=sys.stderr)
                return 1
            target.parent.mkdir(parents=True, exist_ok=True)
            import shutil as _shutil
            _shutil.copy2(source, target)
            print(f"Imported {source} -> {target}")
        elif assets.vocab_tree_present() and not args.force:
            print(f"Already present: {target}")
        else:
            print(f"Downloading {env.VOCAB_TREE_URL}")
            print(f"  -> {target}  (~118 MB, one time)")
            assets.download_vocab_tree(force=args.force)
            print(f"Saved {target}")

    print("\nRun `gs-recon doctor` to confirm everything is in place.")
    return 1 if failed else 0


def cmd_init(args) -> int:
    cfg = Config()
    # Bake in whatever we can detect now, so the file a new user opens is
    # already correct for their machine rather than full of placeholders.
    repo = env.find_lichtfeld_repo()
    if repo:
        cfg.docker.lfs_repo_host = str(repo)

    if args.stdout:
        print(cfg.dump_yaml(), end="")
        return 0

    target = args.output or env.user_config_path()
    if target.exists() and not args.force:
        print(f"error: {target} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    cfg.save(target)
    print(f"Wrote {target}")
    if repo:
        print(f"Detected LichtFeld-Studio checkout: {repo}")
    else:
        print("Could not detect a LichtFeld-Studio checkout -- set docker.lfs_repo_host by hand.")
    return 0


def cmd_plan(args) -> int:
    _, plan = _plan_from_args(args)
    print(plan.describe())
    print(f"\n{len(plan)} step(s) across {len(plan.projects)} project(s).")
    return 0


def cmd_run(args) -> int:
    cfg, plan = _plan_from_args(args)
    if len(plan) == 0:
        print("Nothing to do.", file=sys.stderr)
        return 1

    if not args.skip_doctor and not args.dry_run:
        checks = doctor.run_checks(cfg)
        if doctor.has_blocking_failures(checks):
            print("Environment check failed:\n", file=sys.stderr)
            print(doctor.format_report(checks, color=sys.stderr.isatty()), file=sys.stderr)
            print("\nUse --skip-doctor to run anyway.", file=sys.stderr)
            return 1

    print(plan.describe())
    print(f"\n{len(plan)} step(s) across {len(plan.projects)} project(s).")

    start_index = max(0, args.start_at - 1)
    if start_index >= len(plan):
        print(
            f"error: --start-at {args.start_at} is past the end of a "
            f"{len(plan)}-step plan",
            file=sys.stderr,
        )
        return 1

    if not args.yes and not args.dry_run and sys.stdin.isatty():
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    if not args.dry_run:
        ensure_project_dirs(plan)

    runner = Runner(plan, dry_run=args.dry_run)
    with _interruptible(runner):
        result = runner.run(start_from=start_index)
    return 0 if result.success else 1


@contextlib.contextmanager
def _interruptible(runner: "Runner"):
    """Route Ctrl-C into runner.stop() instead of leaving containers running.

    The child runs in its own session so the terminal's SIGINT never reaches
    it; without this, Ctrl-C unwinds Python while the container keeps training.
    """
    def handle(signum, frame):
        if runner.stopping:
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        print(
            "\n[info] Ctrl-C -- stopping after the current command is "
            "terminated. Press again to abort immediately.",
            file=sys.stderr, flush=True,
        )
        runner.stop()

    try:
        previous = signal.signal(signal.SIGINT, handle)
    except ValueError:      # not the main thread; leave signals alone
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def cmd_gui(args) -> int:
    try:
        from .gui.app import main as gui_main
    except ImportError as exc:
        print(
            f"error: the GUI needs PyQt6, which is not installed ({exc}).\n"
            f"       pip install 'gs-recon-tool[gui]'",
            file=sys.stderr,
        )
        return 1
    return gui_main(config_path=args.config)


def cmd_frames(args) -> int:
    from .tools.frame_extract import main as frames_main
    return frames_main(args.args)


def cmd_prune(args) -> int:
    from .tools.prune_ply import main as prune_main
    return prune_main(args.args)


if __name__ == "__main__":
    raise SystemExit(main())
