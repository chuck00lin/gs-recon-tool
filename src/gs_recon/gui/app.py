"""The gs-recon graphical front end.

Layout, in one sentence: what you are reconstructing lives on the left, how it
gets reconstructed lives in tabs on the right, and what is happening right now
lives along the bottom. Docker plumbing is exiled to its own tab because a new
user should never have to look at it.

The window is a config editor with a Run button. Every control writes into the
same `Config` the CLI consumes, so "Save config" + `gs-recon run -c that.yaml`
reproduces a GUI session exactly.
"""

from __future__ import annotations

import pathlib
import shlex
import sys
import threading
import time
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, doctor, env
from ..config import (
    CAMERA_MODELS,
    CONVERT_FORMATS,
    FILTER_MODES,
    MAPPERS,
    MATCHERS,
    RESOLUTIONS,
    Config,
)
from ..pipeline import (
    Plan,
    ProjectInput,
    build_plan,
    discover_inputs,
    ensure_project_dirs,
    estimate_extracted_frames,
    probe_video,
)
from ..runner import Runner
from ..stages.splat import SplatConfigError
from ..tools.grouping import DEFAULT_GROUP_SIZE, MIN_GROUP_SIZE
from . import theme
from .widgets import GLYPH, TextDialog, form, hint, row

VIDEO_FILTER = "Videos (*.mp4 *.MP4 *.mov *.MOV *.avi *.mkv *.m4v);;All files (*)"

# Curated starting points. Each is a partial config patch applied on top of the
# defaults -- enough to get a usable result without reading any documentation.
PRESETS: dict[str, dict] = {
    "Balanced (default)": {},
    "Fast draft": {
        # group_size tracks the keep rate: ~2 survivors per group is the point
        # where the sharpness test has a choice but the winners cannot clump.
        "frames": {"sample_rate_fps": 10, "filter": {"target": "15%", "group_size": 13}},
        "splat": {"iterations": 7000, "max_cap": 500000, "enable_mip": False},
    },
    "High quality": {
        "frames": {"sample_rate_fps": 20, "filter": {"target": "35%", "group_size": 6}},
        "splat": {"iterations": 50000, "max_cap": 2000000, "enable_mip": True, "ppisp": True},
    },
}


# ---------------------------------------------------------------------------
class RunnerThread(QThread):
    """Runs the pipeline off the GUI thread and feeds output back in batches.

    Batching is not an optimisation, it is what keeps the Stop button usable:
    training prints a progress line per iteration, and one queued signal plus
    one widget update per line saturates the event loop so thoroughly that
    clicks are never processed.
    """

    log = pyqtSignal(str)                    # a batch of lines, newline-joined
    step_started = pyqtSignal(int, int)      # flat index, total
    step_finished = pyqtSignal(int, int)     # flat index, return code
    done = pyqtSignal(bool, int)             # success, failed index (-1 if none)

    FLUSH_INTERVAL = 0.1                     # seconds

    def __init__(self, plan: Plan, start_from: int = 0):
        super().__init__()
        self.plan = plan
        self._start_from = start_from
        self._buffer: list[str] = []
        self._buffer_lock = threading.Lock()
        self._last_flush = 0.0
        self._runner = Runner(
            plan,
            on_log=self._queue_log,
            on_step_start=lambda step, index, total: self.step_started.emit(index, total),
            on_step_end=lambda step, index, rc: self._on_step_end(index, rc),
        )

    # -- logging -----------------------------------------------------------
    def _queue_log(self, line: str) -> None:
        with self._buffer_lock:
            self._buffer.append(line)
        if time.monotonic() - self._last_flush >= self.FLUSH_INTERVAL:
            self.flush_logs()

    def flush_logs(self) -> None:
        """Emit whatever has accumulated. Safe to call from either thread."""
        with self._buffer_lock:
            if not self._buffer:
                return
            batch, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        self.log.emit("\n".join(batch))

    def _on_step_end(self, index: int, rc: int) -> None:
        self.flush_logs()
        self.step_finished.emit(index, rc)

    # -- lifecycle ---------------------------------------------------------
    def stop(self) -> None:
        self._runner.stop()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        result = self._runner.run(start_from=self._start_from)
        self.flush_logs()
        self.done.emit(result.success, -1 if result.failed_index is None else result.failed_index)


# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, config_path: Optional[pathlib.Path] = None):
        super().__init__()
        self.setWindowTitle(f"gs-recon  ·  3D Gaussian Splatting pipeline  ·  {__version__}")
        self.resize(1420, 900)

        self.inputs: list[ProjectInput] = []
        self.config_path: Optional[pathlib.Path] = config_path
        self.thread: Optional[RunnerThread] = None
        self.current_plan: Optional[Plan] = None
        self.failed_index: Optional[int] = None
        self._step_items: list[QTreeWidgetItem] = []
        self._loading = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(180)
        self._refresh_timer.timeout.connect(self._rebuild_tree)

        # Pulls whatever the runner has buffered, so a quiet command still
        # shows its last lines promptly.
        self._log_timer = QTimer(self)
        self._log_timer.setInterval(150)
        self._log_timer.timeout.connect(self._drain_log)

        self._build_ui()
        self._connect_live_math()
        self._apply_config(Config.resolve(config_path))
        self._rebuild_tree()

    # -- construction ------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        root.addWidget(self._build_toolbar())

        work = QSplitter(Qt.Orientation.Horizontal)
        work.setChildrenCollapsible(False)
        work.setHandleWidth(10)
        work.addWidget(self._build_project_panel())
        work.addWidget(self._build_settings_panel())
        work.setSizes([420, 980])

        # Vertical splitter so the log can be dragged to any height instead of
        # permanently eating a third of the window.
        self.vertical_split = QSplitter(Qt.Orientation.Vertical)
        self.vertical_split.setChildrenCollapsible(False)
        self.vertical_split.setHandleWidth(10)
        self.vertical_split.addWidget(work)
        self.vertical_split.addWidget(self._build_progress_panel())
        self.vertical_split.setStretchFactor(0, 1)
        self.vertical_split.setStretchFactor(1, 0)
        self.vertical_split.setSizes([640, 210])
        root.addWidget(self.vertical_split, 1)

        root.addWidget(self._build_footer())

    # ..................................................................
    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.btn_add_videos = QPushButton("Add videos…")
        self.btn_add_videos.setToolTip("Pick one or more video files; each becomes its own project")
        self.btn_add_videos.clicked.connect(self._add_videos)

        self.btn_add_folder = QPushButton("Add folder…")
        self.btn_add_folder.setToolTip(
            "Add a folder of videos, an existing project folder, or (with the "
            "prompt that follows) a parent folder whose subfolders hold videos"
        )
        self.btn_add_folder.clicked.connect(self._add_folder)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_inputs)

        layout.addWidget(self.btn_add_videos)
        layout.addWidget(self.btn_add_folder)
        layout.addWidget(self.btn_clear)

        separator = QLabel("│")
        separator.setObjectName("hint")
        layout.addWidget(separator)

        layout.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS))
        self.preset_combo.setToolTip("Apply a curated set of parameters, then fine-tune below")
        self.preset_combo.activated.connect(self._apply_preset)
        self.preset_combo.setMinimumWidth(170)
        layout.addWidget(self.preset_combo)

        layout.addStretch()

        self.btn_doctor = QPushButton("Check environment")
        self.btn_doctor.setToolTip("Verify Docker, GPU access, images and assets")
        self.btn_doctor.clicked.connect(self._run_doctor)
        layout.addWidget(self.btn_doctor)

        self.btn_load_cfg = QPushButton("Load config…")
        self.btn_load_cfg.clicked.connect(self._load_config)
        layout.addWidget(self.btn_load_cfg)

        self.btn_save_cfg = QPushButton("Save config…")
        self.btn_save_cfg.setToolTip("Write a YAML that `gs-recon run -c <file>` reproduces exactly")
        self.btn_save_cfg.clicked.connect(self._save_config)
        layout.addWidget(self.btn_save_cfg)

        self.btn_copy_cli = QPushButton("Copy CLI command")
        self.btn_copy_cli.setToolTip("Copy the equivalent gs-recon command for running headless")
        self.btn_copy_cli.clicked.connect(self._copy_cli)
        layout.addWidget(self.btn_copy_cli)
        return bar

    # ..................................................................
    def _build_project_panel(self) -> QWidget:
        panel = QGroupBox("Projects")
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        layout.addWidget(hint(
            "Each video becomes a <name>-frames folder next to it. "
            "Existing project folders can be added directly to re-run later stages."
        ))

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Step", ""])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(1, 34)
        header.setStretchLastSection(False)
        layout.addWidget(self.tree, 1)

        self.plan_summary = QLabel("No projects yet")
        self.plan_summary.setObjectName("hint")
        layout.addWidget(self.plan_summary)

        self.btn_preview = QPushButton("Preview full plan…")
        self.btn_preview.setToolTip("See every command that will run, before running it")
        self.btn_preview.clicked.connect(self._preview_plan)
        layout.addWidget(self.btn_preview)
        return panel

    # ..................................................................
    def _build_settings_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        toggles = QWidget()
        toggle_layout = QHBoxLayout(toggles)
        toggle_layout.setContentsMargins(2, 0, 2, 0)
        toggle_layout.setSpacing(18)
        toggle_layout.addWidget(QLabel("Run stages:"))

        self.chk_frames = QCheckBox("① Frames")
        self.chk_sfm = QCheckBox("② SfM")
        self.chk_splat = QCheckBox("③ Splat")
        for check in (self.chk_frames, self.chk_sfm, self.chk_splat):
            check.setChecked(True)
            check.toggled.connect(self._on_stage_toggled)
            toggle_layout.addWidget(check)
        toggle_layout.addStretch()
        layout.addWidget(toggles)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scrollable(self._tab_frames()), "① Frames")
        self.tabs.addTab(self._scrollable(self._tab_sfm()), "② SfM")
        self.tabs.addTab(self._scrollable(self._tab_splat()), "③ Splat")
        self.tabs.addTab(self._scrollable(self._tab_advanced()), "⚙ Advanced")
        layout.addWidget(self.tabs, 1)
        return container

    @staticmethod
    def _scrollable(widget: QWidget) -> QWidget:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    # ..................................................................
    def _tab_frames(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        extraction = QGroupBox("Extraction")
        layout = form(extraction)
        layout.addRow(hint(
            "Sample frames out of each video. More frames means better coverage "
            "but slower matching -- 15 fps on a slow orbit is a good default."
        ))

        self.sp_fps = self._spin(1, 60, 15, "Frames sampled per second of video")
        layout.addRow("Sample rate (fps)", self.sp_fps)

        self.cb_rotation = self._combo(["0", "90", "-90"], "Rotate frames if the camera was held sideways")
        layout.addRow("Rotation (deg)", self.cb_rotation)

        self.sp_quality = self._spin(1, 100, 100, "JPEG quality; keep high, SfM is sensitive to compression artefacts")
        layout.addRow("JPEG quality", self.sp_quality)

        self.cb_resolution = self._combo(list(RESOLUTIONS), "Downscale frames to trade detail for speed")
        self.cb_resolution.currentTextChanged.connect(self._sync_enabled_states)
        layout.addRow("Resolution", self.cb_resolution)

        self.sp_width = self._spin(32, 16384, 1280, "Target width when resolution is 'custom'")
        layout.addRow("Custom width (px)", self.sp_width)

        self.sp_trim_start = self._dspin(0.0, 600.0, 0.0, 0.5, "Seconds to drop from the start of the clip")
        layout.addRow("Trim start (s)", self.sp_trim_start)

        self.sp_trim_end = self._dspin(0.0, 600.0, 0.0, 0.5, "Seconds to drop from the end of the clip")
        layout.addRow("Trim end (s)", self.sp_trim_end)

        self.lbl_extract_math = hint("")
        self.lbl_extract_math.setObjectName("liveMath")
        layout.addRow(self.lbl_extract_math)
        outer.addWidget(extraction)

        filtering = QGroupBox("Sharpness filtering")
        flayout = form(filtering)
        flayout.addRow(hint(
            "Blurry frames poison SfM. Frames are scored by Laplacian variance "
            "and the sharpest are kept, spread evenly so no viewing angle is starved."
        ))

        self.cb_filter_mode = self._combo(
            list(FILTER_MODES),
            "balanced: even coverage · quality: globally sharpest · custom: fixed group count",
        )
        self.cb_filter_mode.currentTextChanged.connect(self._sync_enabled_states)
        flayout.addRow("Mode", self.cb_filter_mode)

        self.ed_filter_target = QLineEdit("20%")
        self.ed_filter_target.setToolTip("A percentage like 20% or an absolute count like 300")
        self.ed_filter_target.textChanged.connect(self._schedule_refresh)
        flayout.addRow("Keep", self.ed_filter_target)

        self.sp_group_size = self._spin(
            MIN_GROUP_SIZE, 500, DEFAULT_GROUP_SIZE,
            "How many consecutive frames compete against each other. The "
            "sharpest of each group survive, so a small group size spreads the "
            "selection evenly and a large one favours sharpness over coverage.",
        )
        flayout.addRow("Frames per group", self.sp_group_size)

        self.sp_filter_groups = self._spin(1, 1000, 20, "Explicit number of groups (custom mode)")
        flayout.addRow("Groups", self.sp_filter_groups)

        # Recomputed on every edit, from the real length of the loaded clips:
        # the group size only means something next to the frame count it
        # divides, and waiting until the run to find that out is too late.
        self.lbl_filter_math = hint("")
        self.lbl_filter_math.setObjectName("liveMath")
        flayout.addRow(self.lbl_filter_math)
        outer.addWidget(filtering)

        outer.addStretch()
        return page

    # ..................................................................
    def _tab_sfm(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        core = QGroupBox("Reconstruction")
        layout = form(core)
        layout.addRow(hint(
            "COLMAP extracts and matches features; GLOMAP solves the global "
            "structure. Both run in the jinwj1996/glomap container."
        ))

        self.cb_camera = self._combo(list(CAMERA_MODELS), "OPENCV models real lens distortion; use PINHOLE only for rectified input")
        layout.addRow("Camera model", self.cb_camera)

        self.cb_matcher = self._combo(
            list(MATCHERS),
            "sequential-loop adds vocabulary-tree loop closure -- the right "
            "choice for a camera orbiting a subject and returning to the start",
        )
        self.cb_matcher.currentTextChanged.connect(self._sync_enabled_states)
        layout.addRow("Matcher", self.cb_matcher)

        self.cb_mapper = self._combo(list(MAPPERS), "GLOMAP is a global solver: much faster than COLMAP's incremental mapper")
        layout.addRow("Mapper", self.cb_mapper)

        self.cb_convert = self._combo(list(CONVERT_FORMATS), "Extra formats written alongside the binary model")
        layout.addRow("Export format", self.cb_convert)
        outer.addWidget(core)

        loop = QGroupBox("Loop closure tuning")
        llayout = form(loop)
        llayout.addRow(hint(
            "Only used by the sequential-loop matcher. The defaults are tuned "
            "for elongated subjects captured in several passes."
        ))
        self.sp_loop_overlap = self._spin(1, 200, 30, "How many temporal neighbours each frame is matched against")
        llayout.addRow("Overlap", self.sp_loop_overlap)
        self.sp_loop_images = self._spin(1, 1000, 100, "Loop-closure candidates retrieved from the vocabulary tree")
        llayout.addRow("Loop candidates", self.sp_loop_images)
        self.sp_loop_period = self._spin(1, 100, 5, "Check for loop closures every N frames")
        llayout.addRow("Loop period", self.sp_loop_period)
        self.loop_group = loop
        outer.addWidget(loop)

        optional = QGroupBox("Optional steps")
        olayout = QVBoxLayout(optional)
        olayout.addWidget(hint(
            "Off by default: GUT mode in the splat stage handles lens "
            "distortion natively, so undistorting first only loses detail."
        ))
        self.chk_undistort = self._check("Undistort images", "Write an undistorted copy into dense/")
        self.chk_reorganize = self._check("Promote undistorted output", "Move dense/ results into images/ and sparse/, backing up the originals")
        self.chk_orient = self._check("Align model orientation", "Rotate the sparse model so its dominant plane is axis-aligned")
        for check in (self.chk_undistort, self.chk_reorganize, self.chk_orient):
            olayout.addWidget(check)
        self.chk_undistort.toggled.connect(self._sync_enabled_states)
        outer.addWidget(optional)

        outer.addStretch()
        return page

    # ..................................................................
    def _tab_splat(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        training = QGroupBox("Training")
        layout = form(training)
        layout.addRow(hint(
            "LichtFeld Studio in GUT mode. Training runs in a throwaway "
            "container with your dataset bind-mounted at its own path."
        ))

        self.sp_iterations = self._spin(1000, 500000, 30000, "More iterations sharpen detail with diminishing returns past ~30k", step=1000)
        layout.addRow("Iterations", self.sp_iterations)

        self.chk_max_cap = self._check("Cap Gaussian count", "Bounds VRAM use; uncheck to let LichtFeld decide")
        self.chk_max_cap.setChecked(True)
        self.chk_max_cap.toggled.connect(self._sync_enabled_states)
        layout.addRow(self.chk_max_cap)

        self.sp_max_cap = self._spin(1000, 20000000, 1000000, "Upper bound on Gaussians (MCMC strategy)", step=100000)
        layout.addRow("Max Gaussians", self.sp_max_cap)

        self.chk_headless = self._check("Headless", "Required when running over SSH or unattended")
        self.chk_headless.setChecked(True)
        layout.addRow(self.chk_headless)

        self.ed_export_ply = QLineEdit("auto")
        self.ed_export_ply.setPlaceholderText("auto")
        self.ed_export_ply.setToolTip(
            "Filename written inside the project's gs/ folder. 'auto' tracks the "
            "iteration count, matching what LichtFeld names its own output."
        )
        self.ed_export_ply.textChanged.connect(self._schedule_refresh)
        layout.addRow("Output PLY", self.ed_export_ply)
        outer.addWidget(training)

        quality = QGroupBox("Quality options")
        qlayout = QVBoxLayout(quality)
        self.chk_ppisp = self._check(
            "Per-camera appearance (--ppisp)",
            "LichtFeld's physically-plausible ISP: learns per-camera exposure and "
            "white balance. Recommended for orbit captures where lighting changes "
            "as you walk around.",
        )
        self.chk_ppisp.setChecked(True)
        self.chk_mip = self._check(
            "Mip anti-aliasing (--enable-mip)",
            "Multi-scale rendering; helps when the subject spans near and far distances.",
        )
        self.chk_mip.setChecked(True)
        self.chk_bilateral = self._check(
            "Bilateral grid (--bilateral-grid)",
            "Alternative appearance correction. Overlaps with --ppisp -- pick one.",
        )
        self.chk_splat_undistort = self._check(
            "Undistort during training (--undistort)",
            "Only if COLMAP did not already undistort and you are not using GUT distortion handling.",
        )
        for check in (self.chk_ppisp, self.chk_mip, self.chk_bilateral, self.chk_splat_undistort):
            qlayout.addWidget(check)
        self.chk_ppisp.toggled.connect(self._sync_enabled_states)
        self.chk_bilateral.toggled.connect(self._sync_enabled_states)
        outer.addWidget(quality)

        prune = QGroupBox("Post-processing")
        player = form(prune)
        player.addRow(hint(
            "Trained splats carry near-transparent Gaussians and an outlier "
            "halo from SfM noise. Pruning them shrinks the file with no visible change."
        ))
        self.chk_prune = self._check("Prune output PLY", "")
        self.chk_prune.setChecked(True)
        self.chk_prune.toggled.connect(self._sync_enabled_states)
        player.addRow(self.chk_prune)

        self.sp_prune_alpha = self._dspin(0.0001, 1.0, 0.005, 0.001, "Drop Gaussians below this opacity", decimals=4)
        player.addRow("Minimum alpha", self.sp_prune_alpha)

        self.sp_prune_pct = self._dspin(50.0, 100.0, 99.0, 0.5, "Radius percentile that defines the keep sphere")
        player.addRow("Radius percentile", self.sp_prune_pct)

        self.sp_prune_margin = self._dspin(1.0, 5.0, 1.0, 0.1, "Multiplier applied to that radius")
        player.addRow("Radius margin", self.sp_prune_margin)
        outer.addWidget(prune)

        outer.addStretch()
        return page

    # ..................................................................
    def _tab_advanced(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        docker = QGroupBox("Docker")
        layout = form(docker)
        layout.addRow(hint(
            "You should not need to touch this. `gs-recon doctor` fills these "
            "in correctly for most machines."
        ))

        self.ed_colmap_image = QLineEdit()
        self.ed_colmap_image.setToolTip("Public image, pulled by `gs-recon setup --pull`")
        self.ed_colmap_image.textChanged.connect(self._schedule_refresh)
        layout.addRow("COLMAP/GLOMAP image", self.ed_colmap_image)

        self.ed_lfs_image = QLineEdit()
        self.ed_lfs_image.setToolTip("Built locally from the LichtFeld-Studio checkout; not pullable")
        self.ed_lfs_image.textChanged.connect(self._schedule_refresh)
        layout.addRow("LichtFeld image", self.ed_lfs_image)

        self.ed_lfs_repo = QLineEdit()
        self.ed_lfs_repo.setPlaceholderText("auto-detected")
        self.ed_lfs_repo.setToolTip(
            "Host folder holding build/LichtFeld-Studio. The image ships no "
            "binary -- it lives in this checkout."
        )
        self.ed_lfs_repo.textChanged.connect(self._schedule_refresh)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_lfs_repo)
        layout.addRow("LichtFeld checkout", row(self.ed_lfs_repo, browse, stretch_last=False))

        self.ed_lfs_container = QLineEdit()
        self.ed_lfs_container.setToolTip(
            "Container path the checkout is mounted at. Must match the path the "
            "binary was built for -- its RUNPATH is absolute. 'auto' probes the image."
        )
        self.ed_lfs_container.textChanged.connect(self._schedule_refresh)
        layout.addRow("Container mount path", self.ed_lfs_container)

        self.ed_gpus = QLineEdit()
        self.ed_gpus.setToolTip("Passed to docker --gpus: 'all', 'device=0', 'device=0,1'")
        self.ed_gpus.textChanged.connect(self._schedule_refresh)
        layout.addRow("GPUs", self.ed_gpus)

        self.ed_extra_run = QLineEdit()
        self.ed_extra_run.setPlaceholderText("e.g. --shm-size=8g")
        self.ed_extra_run.setToolTip("Extra arguments inserted into every docker run")
        self.ed_extra_run.textChanged.connect(self._schedule_refresh)
        layout.addRow("Extra docker args", self.ed_extra_run)

        self.ed_splat_extra = QLineEdit()
        self.ed_splat_extra.setPlaceholderText("e.g. --strategy mcmc --sh-degree 2")
        self.ed_splat_extra.setToolTip("Appended verbatim to the LichtFeld training command")
        self.ed_splat_extra.textChanged.connect(self._schedule_refresh)
        layout.addRow("Extra LichtFeld args", self.ed_splat_extra)
        outer.addWidget(docker)

        info = QGroupBox("This machine")
        ilayout = QVBoxLayout(info)
        self.lbl_machine = QLabel()
        self.lbl_machine.setObjectName("hint")
        self.lbl_machine.setWordWrap(True)
        self.lbl_machine.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ilayout.addWidget(self.lbl_machine)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_machine_info)
        ilayout.addWidget(refresh, 0, Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(info)
        self._refresh_machine_info()

        outer.addStretch()
        return page

    # ..................................................................
    def _build_progress_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("idle")
        top.addWidget(self.progress, 1)

        self.btn_toggle_log = QPushButton("Hide log")
        self.btn_toggle_log.setCheckable(True)
        self.btn_toggle_log.setChecked(True)
        self.btn_toggle_log.clicked.connect(self._toggle_log)
        top.addWidget(self.btn_toggle_log)
        layout.addLayout(top)

        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)   # bounded so long runs cannot eat RAM
        self.log.setMinimumHeight(90)
        layout.addWidget(self.log, 1)
        return panel

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.status_pill = QLabel("Idle")
        self.status_pill.setObjectName("statusPill")
        layout.addWidget(self.status_pill)
        layout.addStretch()

        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumWidth(150)
        self.btn_start.clicked.connect(self._start)
        layout.addWidget(self.btn_start)

        self.btn_resume = QPushButton("Resume from failure")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self._resume)
        layout.addWidget(self.btn_resume)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        layout.addWidget(self.btn_stop)
        return footer

    # -- widget factories --------------------------------------------------
    def _spin(self, lo: int, hi: int, value: int, tip: str, *, step: int = 1) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(lo, hi)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setToolTip(tip)
        widget.setMinimumWidth(140)
        widget.valueChanged.connect(self._schedule_refresh)
        return widget

    def _dspin(self, lo: float, hi: float, value: float, step: float, tip: str,
               *, decimals: int = 2) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(lo, hi)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setToolTip(tip)
        widget.setMinimumWidth(140)
        widget.valueChanged.connect(self._schedule_refresh)
        return widget

    def _combo(self, items: list[str], tip: str) -> QComboBox:
        widget = QComboBox()
        widget.addItems(items)
        widget.setToolTip(tip)
        widget.setMinimumWidth(200)
        widget.currentTextChanged.connect(self._schedule_refresh)
        return widget

    def _check(self, text: str, tip: str) -> QCheckBox:
        widget = QCheckBox(text)
        if tip:
            widget.setToolTip(tip)
        widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        widget.toggled.connect(self._schedule_refresh)
        return widget

    # -- config <-> widgets ------------------------------------------------
    def _apply_config(self, cfg: Config) -> None:
        self._loading = True
        try:
            f, s, sp, d = cfg.frames, cfg.sfm, cfg.splat, cfg.docker

            self.chk_frames.setChecked(f.enabled)
            self.chk_sfm.setChecked(s.enabled)
            self.chk_splat.setChecked(sp.enabled)

            self.sp_fps.setValue(f.sample_rate_fps)
            self.cb_rotation.setCurrentText(str(f.rotation))
            self.sp_quality.setValue(f.jpeg_quality)
            self.cb_resolution.setCurrentText(f.resolution)
            self.sp_width.setValue(f.custom_width)
            self.sp_trim_start.setValue(f.trim_start)
            self.sp_trim_end.setValue(f.trim_end)
            self.cb_filter_mode.setCurrentText(f.filter.mode)
            self.ed_filter_target.setText(str(f.filter.target))
            self.sp_group_size.setValue(f.filter.group_size)
            self.sp_filter_groups.setValue(f.filter.groups)

            self.cb_camera.setCurrentText(s.camera_model)
            self.cb_matcher.setCurrentText(s.matcher)
            self.cb_mapper.setCurrentText(s.mapper)
            self.cb_convert.setCurrentText(s.convert)
            self.chk_undistort.setChecked(s.undistort)
            self.chk_reorganize.setChecked(s.reorganize)
            self.chk_orient.setChecked(s.orient)
            self.sp_loop_overlap.setValue(s.loop_overlap)
            self.sp_loop_images.setValue(s.loop_detection_num_images)
            self.sp_loop_period.setValue(s.loop_detection_period)

            self.sp_iterations.setValue(sp.iterations)
            self.chk_headless.setChecked(sp.headless)
            self.chk_max_cap.setChecked(sp.max_cap is not None)
            if sp.max_cap:
                self.sp_max_cap.setValue(sp.max_cap)
            self.chk_ppisp.setChecked(sp.ppisp)
            self.chk_mip.setChecked(sp.enable_mip)
            self.chk_bilateral.setChecked(sp.bilateral_grid)
            self.chk_splat_undistort.setChecked(sp.undistort)
            self.ed_export_ply.setText(sp.export_ply)
            self.ed_splat_extra.setText(sp.extra_args)
            self.chk_prune.setChecked(sp.prune.enabled)
            self.sp_prune_alpha.setValue(sp.prune.alpha)
            self.sp_prune_pct.setValue(sp.prune.percentile)
            self.sp_prune_margin.setValue(sp.prune.margin)

            self.ed_colmap_image.setText(d.colmap_image)
            self.ed_lfs_image.setText(d.lfs_image)
            self.ed_lfs_repo.setText(d.lfs_repo_host)
            self.ed_lfs_container.setText(d.lfs_repo_container)
            self.ed_gpus.setText(d.gpus)
            self.ed_extra_run.setText(" ".join(d.extra_run_args))
        finally:
            self._loading = False
        self._sync_enabled_states()

    def _config_from_ui(self) -> Config:
        cfg = Config()
        f, s, sp, d = cfg.frames, cfg.sfm, cfg.splat, cfg.docker

        f.enabled = self.chk_frames.isChecked()
        f.sample_rate_fps = self.sp_fps.value()
        f.rotation = int(self.cb_rotation.currentText())
        f.jpeg_quality = self.sp_quality.value()
        f.resolution = self.cb_resolution.currentText()
        f.custom_width = self.sp_width.value()
        f.trim_start = self.sp_trim_start.value()
        f.trim_end = self.sp_trim_end.value()
        f.filter.mode = self.cb_filter_mode.currentText()
        f.filter.target = self.ed_filter_target.text().strip() or "20%"
        f.filter.group_size = self.sp_group_size.value()
        f.filter.groups = self.sp_filter_groups.value()

        s.enabled = self.chk_sfm.isChecked()
        s.camera_model = self.cb_camera.currentText()
        s.matcher = self.cb_matcher.currentText()
        s.mapper = self.cb_mapper.currentText()
        s.convert = self.cb_convert.currentText()
        s.undistort = self.chk_undistort.isChecked()
        s.reorganize = self.chk_reorganize.isChecked()
        s.orient = self.chk_orient.isChecked()
        s.loop_overlap = self.sp_loop_overlap.value()
        s.loop_detection_num_images = self.sp_loop_images.value()
        s.loop_detection_period = self.sp_loop_period.value()

        sp.enabled = self.chk_splat.isChecked()
        sp.iterations = self.sp_iterations.value()
        sp.headless = self.chk_headless.isChecked()
        sp.max_cap = self.sp_max_cap.value() if self.chk_max_cap.isChecked() else None
        sp.ppisp = self.chk_ppisp.isChecked()
        sp.enable_mip = self.chk_mip.isChecked()
        sp.bilateral_grid = self.chk_bilateral.isChecked()
        sp.undistort = self.chk_splat_undistort.isChecked()
        sp.export_ply = self.ed_export_ply.text().strip() or "auto"
        sp.extra_args = self.ed_splat_extra.text().strip()
        sp.prune.enabled = self.chk_prune.isChecked()
        sp.prune.alpha = self.sp_prune_alpha.value()
        sp.prune.percentile = self.sp_prune_pct.value()
        sp.prune.margin = self.sp_prune_margin.value()

        d.colmap_image = self.ed_colmap_image.text().strip() or "jinwj1996/glomap"
        d.lfs_image = self.ed_lfs_image.text().strip() or "lichtfeld-studio:latest"
        d.lfs_repo_host = self.ed_lfs_repo.text().strip()
        d.lfs_repo_container = self.ed_lfs_container.text().strip() or "auto"
        d.gpus = self.ed_gpus.text().strip()
        try:
            d.extra_run_args = shlex.split(self.ed_extra_run.text().strip())
        except ValueError:
            d.extra_run_args = []
        return cfg

    def _connect_live_math(self) -> None:
        """Recompute the live arithmetic whenever an input to it changes."""
        for spin in (self.sp_fps, self.sp_group_size, self.sp_filter_groups):
            spin.valueChanged.connect(self._schedule_refresh)
        for dspin in (self.sp_trim_start, self.sp_trim_end):
            dspin.valueChanged.connect(self._schedule_refresh)
        self.cb_filter_mode.currentTextChanged.connect(self._schedule_refresh)

    def _sync_enabled_states(self) -> None:
        """Grey out controls that cannot affect the current configuration."""
        self.sp_width.setEnabled(self.cb_resolution.currentText() == "custom")
        mode = self.cb_filter_mode.currentText()
        self.sp_group_size.setEnabled(mode == "balanced")
        self.sp_filter_groups.setEnabled(mode == "custom")

        self.loop_group.setEnabled(self.cb_matcher.currentText() == "sequential-loop")
        self.chk_reorganize.setEnabled(self.chk_undistort.isChecked())
        if not self.chk_undistort.isChecked() and self.chk_reorganize.isChecked():
            self.chk_reorganize.setChecked(False)

        self.sp_max_cap.setEnabled(self.chk_max_cap.isChecked())
        for widget in (self.sp_prune_alpha, self.sp_prune_pct, self.sp_prune_margin):
            widget.setEnabled(self.chk_prune.isChecked())

        # --ppisp and --bilateral-grid both correct appearance; enabling both
        # wastes capacity, so make the conflict visible rather than silent.
        conflict = self.chk_ppisp.isChecked() and self.chk_bilateral.isChecked()
        self.chk_bilateral.setStyleSheet("color: #b45309;" if conflict else "")
        self.chk_bilateral.setToolTip(
            "Conflicts with --ppisp, which is also enabled. Turn one off."
            if conflict else
            "Alternative appearance correction. Overlaps with --ppisp -- pick one."
        )

        self._schedule_refresh()

    def _on_stage_toggled(self) -> None:
        self.tabs.setTabEnabled(0, self.chk_frames.isChecked())
        self.tabs.setTabEnabled(1, self.chk_sfm.isChecked())
        self.tabs.setTabEnabled(2, self.chk_splat.isChecked())
        self._schedule_refresh()

    # -- inputs ------------------------------------------------------------
    def _add_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select video files", str(pathlib.Path.home()), VIDEO_FILTER
        )
        if files:
            self._add_paths(files, recursive=False)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select a folder", str(pathlib.Path.home())
        )
        if not folder:
            return
        # Try the simple interpretation first; only offer the recursive scan if
        # the folder itself yields nothing, so the common case stays one click.
        found, _ = discover_inputs([folder], recursive=False)
        if found:
            self._add_paths([folder], recursive=False)
            return
        answer = QMessageBox.question(
            self, "Scan subfolders?",
            f"No videos or project markers directly inside:\n{folder}\n\n"
            f"Scan its subfolders for videos?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._add_paths([folder], recursive=True)

    def _add_paths(self, paths: list[str], *, recursive: bool) -> None:
        found, warnings = discover_inputs(paths, recursive=recursive)
        for warning in warnings:
            self._append_log(f"[warning] {warning}")
        existing = {item.project for item in self.inputs}
        added = 0
        for item in found:
            if item.project not in existing:
                self.inputs.append(item)
                existing.add(item.project)
                added += 1
        if added:
            self._append_log(f"[info] Added {added} project(s).")
        elif found:
            self._append_log("[info] Those projects were already in the list.")
        else:
            QMessageBox.information(
                self, "Nothing added",
                "No videos or existing project folders were found in that selection.",
            )
        self._rebuild_tree()

    def _clear_inputs(self) -> None:
        self.inputs.clear()
        self.failed_index = None
        self.btn_resume.setEnabled(False)
        self._rebuild_tree()

    # -- plan / tree -------------------------------------------------------
    def _schedule_refresh(self) -> None:
        if self._loading:
            return
        self._refresh_timer.start()

    def _current_plan(self) -> tuple[Optional[Plan], str]:
        cfg = self._config_from_ui()
        problems = cfg.validate()
        if problems:
            return None, "\n".join(f"• {p}" for p in problems)
        try:
            return build_plan(cfg, self.inputs), ""
        except SplatConfigError as exc:
            return None, str(exc)

    def _rebuild_tree(self) -> None:
        plan, error = self._current_plan()
        self.current_plan = plan
        self._refresh_estimates(self._config_from_ui())
        self.tree.clear()
        self._step_items = []

        if not self.inputs:
            self.plan_summary.setText("No projects yet — add a video or a project folder to begin.")
            self.btn_start.setEnabled(False)
            return

        if plan is None:
            self.plan_summary.setText("⚠ " + error.splitlines()[0])
            self.plan_summary.setToolTip(error)
            self.btn_start.setEnabled(False)
            placeholder = QTreeWidgetItem(self.tree, ["Configuration problem", ""])
            placeholder.setToolTip(0, error)
            return

        self.plan_summary.setToolTip("")
        monospace = QFont("DejaVu Sans Mono")
        for project_plan in plan.projects:
            parent = QTreeWidgetItem(self.tree, [project_plan.project.name, ""])
            parent.setToolTip(0, str(project_plan.project))
            parent.setExpanded(True)
            for step in project_plan.steps:
                child = QTreeWidgetItem(parent, [step.label, GLYPH["pending"]])
                child.setToolTip(0, f"{step.note}\n\n$ {step.display()}" if step.note else step.display())
                child.setFont(1, monospace)
                self._step_items.append(child)

        total = len(plan)
        self.plan_summary.setText(
            f"{total} step(s) across {len(plan.projects)} project(s)"
            if total else "Every stage is disabled — nothing would run."
        )
        self.btn_start.setEnabled(total > 0 and self.thread is None)

    # -- live arithmetic ---------------------------------------------------
    def _frame_totals(self, cfg: Config) -> tuple[Optional[int], int, int, float]:
        """(estimated frames, inputs measured, inputs we could not, seconds)."""
        total = 0
        measured = 0
        unknown = 0
        seconds = 0.0
        for item in self.inputs:
            if item.video is not None and cfg.frames.enabled:
                count = estimate_extracted_frames(cfg.frames, item.video)
                stats = probe_video(item.video)
                if stats is not None:
                    seconds += max(
                        0.0,
                        stats.duration - cfg.frames.trim_start - cfg.frames.trim_end,
                    )
            else:
                # An existing project brings its own frames; count what is
                # already on disk so the same arithmetic still applies.
                images = item.project / "images"
                count = sum(
                    1 for entry in images.glob("*")
                    if entry.suffix.lower() in (".jpg", ".jpeg", ".png")
                ) if images.is_dir() else None
                count = count or None
            if count:
                total += count
                measured += 1
            else:
                unknown += 1
        return (total or None), measured, unknown, seconds

    def _refresh_estimates(self, cfg: Config) -> None:
        estimate, measured, unknown, seconds = self._frame_totals(cfg)

        if not self.inputs:
            self.lbl_extract_math.setText(
                "Add a clip to see how many frames these settings produce."
            )
        elif estimate is None:
            self.lbl_extract_math.setText("Could not read the length of the input(s).")
        else:
            length = f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"
            footage = f" from {length} of footage" if seconds else ""
            note = f" (+{unknown} unreadable)" if unknown else ""
            self.lbl_extract_math.setText(
                f"≈ {estimate:,} frames{footage} across {measured} input(s){note}, "
                f"at {cfg.frames.sample_rate_fps} fps."
            )

        self.lbl_filter_math.setText(cfg.frames.filter.describe(estimate))

    def _set_step_status(self, index: int, status: str) -> None:
        if 0 <= index < len(self._step_items):
            self._step_items[index].setText(1, GLYPH[status])
            self._step_items[index].setSelected(status == "running")
            if status == "running":
                self.tree.scrollToItem(self._step_items[index])

    # -- actions -----------------------------------------------------------
    def _apply_preset(self) -> None:
        name = self.preset_combo.currentText()
        patch = PRESETS.get(name, {})
        cfg = Config()
        # Preserve the user's stage toggles and Docker settings; a preset is
        # about quality/speed, not about where their containers live.
        current = self._config_from_ui()
        cfg.docker = current.docker
        cfg.frames.enabled = current.frames.enabled
        cfg.sfm.enabled = current.sfm.enabled
        cfg.splat.enabled = current.splat.enabled
        _deep_update(cfg, patch)
        self._apply_config(cfg)
        self._append_log(f"[info] Applied preset: {name}")
        self._rebuild_tree()

    def _browse_lfs_repo(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select the LichtFeld-Studio checkout", str(pathlib.Path.home())
        )
        if folder:
            self.ed_lfs_repo.setText(folder)

    def _refresh_machine_info(self) -> None:
        memory = env.host_memory()
        gpus = env.gpu_names() or ["(no GPU detected)"]
        parts = [
            f"Docker: {env.docker_version() or 'not available'}",
            f"GPU: {'; '.join(gpus)}",
        ]
        if memory:
            parts.append(
                f"RAM: {memory.get('mem_total_gb', 0):.1f} GB total, "
                f"{memory.get('mem_available_gb', 0):.1f} GB available"
            )
        parts.append(f"Vocabulary tree: {env.vocab_tree_path()}")
        parts.append(f"Config: {self.config_path or env.user_config_path()}")
        self.lbl_machine.setText("\n".join(parts))

    def _run_doctor(self) -> None:
        self.btn_doctor.setEnabled(False)
        self.btn_doctor.setText("Checking…")
        QApplication.processEvents()
        try:
            checks = doctor.run_checks(self._config_from_ui())
            report = doctor.format_report(checks, color=False)
        finally:
            self.btn_doctor.setEnabled(True)
            self.btn_doctor.setText("Check environment")
        TextDialog(
            "Environment check", report, self,
            subtitle="Anything marked -> has a fix you can copy and run.",
            width=880, height=520,
        ).exec()

    def _preview_plan(self) -> None:
        plan, error = self._current_plan()
        if plan is None:
            QMessageBox.warning(self, "Cannot build a plan", error)
            return
        if not len(plan):
            QMessageBox.information(self, "Nothing to run", "No steps are enabled.")
            return
        TextDialog(
            "Execution plan", plan.describe(), self,
            subtitle=f"{len(plan)} step(s). These are the exact commands that will run.",
        ).exec()

    def _save_config(self) -> None:
        cfg = self._config_from_ui()
        problems = cfg.validate()
        if problems:
            QMessageBox.warning(
                self, "Config is not valid",
                "Fix these first:\n\n" + "\n".join(f"• {p}" for p in problems),
            )
            return
        default = str(self.config_path or env.user_config_path())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", default, "YAML (*.yaml *.yml);;All files (*)"
        )
        if not path:
            return
        saved = cfg.save(pathlib.Path(path))
        self.config_path = saved
        self._append_log(f"[info] Config saved to {saved}")
        self._refresh_machine_info()

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config", str(env.user_config_path().parent),
            "YAML (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        try:
            cfg = Config.load(pathlib.Path(path))
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not load config", str(exc))
            return
        self.config_path = pathlib.Path(path)
        self._apply_config(cfg)
        self._append_log(f"[info] Config loaded from {path}")
        self._rebuild_tree()
        self._refresh_machine_info()

    def _copy_cli(self) -> None:
        if not self.inputs:
            QMessageBox.information(
                self, "No projects", "Add at least one video or project folder first."
            )
            return
        cfg = self._config_from_ui()
        stages = [
            name for name, enabled in (
                ("frames", cfg.frames.enabled),
                ("sfm", cfg.sfm.enabled),
                ("splat", cfg.splat.enabled),
            ) if enabled
        ]
        sources = [
            str(item.video) if item.video else str(item.project) for item in self.inputs
        ]
        argv = ["gs-recon", "run", *sources]
        if self.config_path:
            argv += ["-c", str(self.config_path)]
        if len(stages) < 3:
            argv += ["--only", ",".join(stages)]
        argv.append("--yes")
        command = shlex.join(argv)

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(command)
        note = (
            "" if self.config_path else
            "\n\nNote: you have not saved a config yet, so this command uses "
            "the defaults rather than your current settings. Save a config "
            "first and copy again to carry them over."
        )
        QMessageBox.information(
            self, "Copied to clipboard", f"{command}{note}"
        )

    # -- run control -------------------------------------------------------
    def _start(self) -> None:
        plan, error = self._current_plan()
        if plan is None:
            QMessageBox.warning(self, "Cannot start", error)
            return
        if not len(plan):
            QMessageBox.information(self, "Nothing to run", "No steps are enabled.")
            return

        checks = doctor.run_checks(self._config_from_ui())
        if doctor.has_blocking_failures(checks):
            report = doctor.format_report(checks, color=False)
            answer = QMessageBox.question(
                self, "Environment check failed",
                "Some prerequisites are missing. Start anyway?\n\n"
                + "\n".join(
                    f"✗ {c.name}: {c.detail}" for c in checks if c.status == doctor.FAIL
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                TextDialog("Environment check", report, self).exec()
                return

        ensure_project_dirs(plan)
        self.current_plan = plan
        self.failed_index = None
        for index in range(len(self._step_items)):
            self._set_step_status(index, "pending")
        self.log.clear()
        self._append_log(f"=== gs-recon {__version__} — starting {len(plan)} step(s) ===")
        self._launch(plan, start_from=0)

    def _resume(self) -> None:
        if self.current_plan is None or self.failed_index is None:
            return
        self._append_log(f"\n=== Resuming from step {self.failed_index + 1} ===")
        self._launch(self.current_plan, start_from=self.failed_index)

    def _launch(self, plan: Plan, *, start_from: int) -> None:
        self.progress.setRange(0, max(1, len(plan)))
        self.progress.setValue(start_from)
        self._set_running(True)

        self.thread = RunnerThread(plan, start_from=start_from)
        self.thread.log.connect(self._append_log)
        self.thread.step_started.connect(self._on_step_started)
        self.thread.step_finished.connect(self._on_step_finished)
        self.thread.done.connect(self._on_done)
        self.thread.start()
        self._log_timer.start()

    def _stop(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.btn_stop.setEnabled(False)
            self.status_pill.setText("Stopping…")
            self.thread.stop()

    def _on_step_started(self, index: int, total: int) -> None:
        self._set_step_status(index, "running")
        step = self.current_plan.steps[index] if self.current_plan else None
        label = step.label if step else ""
        project = step.project.name if step and step.project else ""
        self.progress.setValue(index)
        self.progress.setFormat(f"{index + 1}/{total}  ·  %p%")
        self.status_pill.setText(f"{label}  ·  {project}")

    def _on_step_finished(self, index: int, rc: int) -> None:
        self._set_step_status(index, "done" if rc == 0 else "failed")
        if rc == 0:
            self.progress.setValue(index + 1)

    def _drain_log(self) -> None:
        if self.thread is not None:
            self.thread.flush_logs()

    def _on_done(self, success: bool, failed_index: int) -> None:
        self._log_timer.stop()
        self._drain_log()
        self._set_running(False)
        self.thread = None
        if success:
            self.status_pill.setText("✓ Completed")
            self.progress.setValue(self.progress.maximum())
            self.progress.setFormat("done  ·  100%")
            self.failed_index = None
            self.btn_resume.setEnabled(False)
        else:
            self.status_pill.setText("✕ Stopped")
            self.progress.setFormat("stopped  ·  %p%")
            if failed_index >= 0:
                self.failed_index = failed_index
                self.btn_resume.setEnabled(True)
                self.btn_resume.setText(f"Resume from step {failed_index + 1}")
                self._append_log(
                    f"[info] Fix the problem above, then press "
                    f"'Resume from step {failed_index + 1}'."
                )

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_resume.setEnabled(not running and self.failed_index is not None)
        for widget in (
            self.btn_add_videos, self.btn_add_folder, self.btn_clear,
            self.preset_combo, self.btn_load_cfg, self.tabs,
            self.chk_frames, self.chk_sfm, self.chk_splat,
        ):
            widget.setEnabled(not running)
        if running:
            self.status_pill.setText("Running…")

    # -- misc --------------------------------------------------------------
    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _toggle_log(self) -> None:
        visible = self.btn_toggle_log.isChecked()
        self.log.setVisible(visible)
        self.btn_toggle_log.setText("Hide log" if visible else "Show log")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.thread is not None and self.thread.isRunning():
            answer = QMessageBox.question(
                self, "A run is in progress",
                "Stop the running pipeline and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.thread.stop()
            self.thread.wait(5000)
        event.accept()


def _deep_update(cfg: Config, patch: dict) -> None:
    """Apply a nested preset patch onto a Config in place."""
    for section, values in patch.items():
        target = getattr(cfg, section)
        for key, value in values.items():
            if isinstance(value, dict):
                nested = getattr(target, key)
                for nested_key, nested_value in value.items():
                    setattr(nested, nested_key, nested_value)
            else:
                setattr(target, key, value)


# ---------------------------------------------------------------------------
def main(config_path: Optional[pathlib.Path] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("gs-recon")
    app.setStyle("Fusion")

    base = app.palette().window().color()
    colors = theme.palette_for(base.lightness() < 128)
    theme.apply_palette(app, colors)
    app.setStyleSheet(theme.stylesheet(colors))

    window = MainWindow(config_path=config_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
