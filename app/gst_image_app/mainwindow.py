"""Main application window for interactive microstructure analysis."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, QThreadPool, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QColor,
    QImage,
    QKeySequence,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from gst_image.analysis import (
    ModelPack,
    build_analysis_mask,
    classify_regions,
    compute_project_fractions,
    create_measurement,
    evaluate_particle_grouping,
    measure_particles,
    particle_group_statistics,
    run_cellpose_inference,
    run_model_inference,
    segment_particles,
    suggest_specimen_mask,
)
from gst_image.analysis.groups import ParticleGroupingResult
from gst_image.analysis.particles import merge_particle_labels, split_particle_by_line
from gst_image.analysis.preprocess import to_gray
from gst_image.analysis.regions import REGION_CLASSIFICATION_MAX_PIXELS
from gst_image.export import create_particle_group_overlay, export_analysis
from gst_image.image_io import (
    load_image,
    load_region,
    load_scaled_image,
    sha256_file,
)
from gst_image.models import (
    ROI,
    AnalysisRun,
    Calibration,
    CellposeInferenceRecipe,
    CellposeInferenceRun,
    ClassDefinition,
    EditEvent,
    ModelInferenceRun,
    MorphologicalGradient,
    MorphologicalInput,
    ParticleGrouping,
    ParticlePolarity,
    Point,
    ProjectManifest,
    RegionClassifierRecipe,
    ROIKind,
    SegmentationLayer,
    SegmentationMethod,
    SegmentationRecipe,
    ThresholdMethod,
    TrainingStroke,
)
from gst_image.project import (
    load_project,
    relink_source,
    resolve_source,
    save_project,
    validate_project,
)
from gst_image_app.canvas import ImageCanvas
from gst_image_app.particle_grouping import ParticleGroupingPanel
from gst_image_app.workers import FunctionWorker


def _analyze_source(path, rois, include_ids, recipe, calibration, *, progress, cancelled):
    source = load_image(path, color=recipe.channel != "gray")
    gray = to_gray(source, recipe.channel)
    progress(0.01, "Building specimen mask")
    specimen = suggest_specimen_mask(gray)
    domain = build_analysis_mask(
        gray.shape,
        rois,
        specimen,
        set(include_ids),
    )
    result = segment_particles(
        source,
        domain,
        recipe,
        calibration,
        progress=progress,
        cancelled=cancelled,
    )
    return gray, domain, result, (0, 0, gray.shape[1], gray.shape[0])


def _analyze_model_source(path, pack_path, recipe, calibration, *, progress, cancelled):
    """Run a separately installed model pack in the same worker path as particles."""
    source = load_image(path, color=True)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    progress(0.01, "Building model analysis domain")
    domain = build_analysis_mask(gray.shape, specimen_mask=suggest_specimen_mask(gray))
    pack = ModelPack.open(pack_path)
    result = run_model_inference(
        source,
        domain,
        pack,
        recipe,
        calibration,
        progress=progress,
        cancelled=cancelled,
    )
    return gray, domain, result, pack, recipe


def _analyze_cellpose_source(
    path,
    rois,
    include_ids,
    recipe,
    calibration,
    allow_model_download,
    *,
    progress,
    cancelled,
):
    """Run local Cellpose at source resolution inside the Analysis boxes only."""
    source = load_image(path, color=True)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    progress(0.01, "Building Cellpose analysis domain")
    boxes = _cellpose_analysis_boxes(rois, include_ids)
    if not boxes:
        raise ValueError(
            "Cellpose runs inside Analysis boxes only. Draw at least one Analysis box on the "
            "overview before running Cellpose-SAM."
        )
    scoped = boxes + [roi for roi in rois if roi.kind == ROIKind.EXCLUDE]
    domain = build_analysis_mask(
        gray.shape,
        scoped,
        suggest_specimen_mask(gray),
        {roi.id for roi in boxes},
    )
    result = run_cellpose_inference(
        source,
        domain,
        recipe,
        calibration,
        regions=[roi_bounds(gray.shape, roi) for roi in boxes],
        allow_model_download=allow_model_download,
        progress=progress,
        cancelled=cancelled,
    )
    return gray, domain, result, recipe


def _cellpose_analysis_boxes(rois, include_ids) -> list[ROI]:
    """Cellpose is scoped to Analysis boxes, narrowed further by an explicit selection."""
    wanted = set(include_ids or ())
    return [
        roi
        for roi in rois
        if roi.kind == ROIKind.ANALYSIS_BOX and (not wanted or roi.id in wanted)
    ]


def _cuda_is_available() -> bool:
    """Report CUDA readiness without making Torch a hard requirement of the GUI."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def roi_bounds(shape: tuple[int, int], roi: ROI) -> tuple[int, int, int, int]:
    """Clamp an ROI's pixel bounding box to the image as (left, top, right, bottom)."""
    height, width = shape
    xs = [point.x for point in roi.points]
    ys = [point.y for point in roi.points]
    left = min(width - 1, max(0, int(np.floor(min(xs)))))
    top = min(height - 1, max(0, int(np.floor(min(ys)))))
    right = min(width, int(np.ceil(max(xs))) + 1)
    bottom = min(height, int(np.ceil(max(ys))) + 1)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _analyze_source_region(
    path,
    bounds,
    rois,
    include_ids,
    recipe,
    calibration,
    *,
    progress,
    cancelled,
):
    """Load and analyze only one native-resolution source region."""
    source, actual_bounds = load_region(
        path,
        bounds,
        color=recipe.channel != "gray",
        scale_percent=100,
    )
    left, top, _, _ = actual_bounds
    local_rois = [
        roi.model_copy(
            update={
                "points": [
                    Point(x=point.x - left, y=point.y - top)
                    for point in roi.points
                ]
            }
        )
        for roi in rois
    ]
    gray = to_gray(source, recipe.channel)
    progress(0.01, "Building selected-ROI specimen mask")
    specimen = suggest_specimen_mask(gray)
    domain = build_analysis_mask(
        gray.shape,
        local_rois,
        specimen,
        set(include_ids),
    )
    result = segment_particles(
        source,
        domain,
        recipe,
        calibration,
        progress=progress,
        cancelled=cancelled,
    )
    return gray, domain, result, actual_bounds


def _analyze_preview(
    preview,
    full_size,
    rois,
    include_ids,
    recipe,
    calibration,
    *,
    progress,
    cancelled,
):
    """Scale pixel-domain settings and ROIs for a fast, non-persistent preview run."""
    height, width = preview.shape[:2]
    scale_x = width / full_size[0]
    scale_y = height / full_size[1]
    scale = (scale_x * scale_y) ** 0.5

    def odd_scaled(value: int) -> int:
        result = max(3, round(value * scale))
        return result if result % 2 else result + 1

    scaled_recipe = recipe.model_copy(
        update={
            "rolling_ball_radius_px": max(3, round(recipe.rolling_ball_radius_px * scale)),
            "sauvola_window_px": odd_scaled(recipe.sauvola_window_px),
            "gaussian_block_px": odd_scaled(recipe.gaussian_block_px),
            "gaussian_blur_sigma": recipe.gaussian_blur_sigma * scale,
            "open_radius_px": max(0, round(recipe.open_radius_px * scale)),
            "close_radius_px": max(0, round(recipe.close_radius_px * scale)),
            "min_particle_area_px": max(0, round(recipe.min_particle_area_px * scale**2)),
            "max_particle_area_px": (
                max(1, round(recipe.max_particle_area_px * scale**2))
                if recipe.max_particle_area_px is not None
                else None
            ),
            "watershed_min_distance_px": max(
                1, round(recipe.watershed_min_distance_px * scale)
            ),
            "morphological_gradient_radius_px": max(
                1, round(recipe.morphological_gradient_radius_px * scale)
            ),
            "tile_size_px": max(256, round(recipe.tile_size_px * scale)),
        }
    )
    scaled_rois = [
        roi.model_copy(
            update={
                "points": [
                    Point(x=point.x * scale_x, y=point.y * scale_y)
                    for point in roi.points
                ]
            }
        )
        for roi in rois
    ]
    preview_calibration = (
        calibration.model_copy(update={"mm_per_pixel": calibration.mm_per_pixel / scale})
        if calibration
        else None
    )
    gray = to_gray(preview, scaled_recipe.channel)
    specimen = suggest_specimen_mask(gray)
    domain = build_analysis_mask(
        gray.shape,
        scaled_rois,
        specimen,
        set(include_ids),
    )
    return segment_particles(
        preview,
        domain,
        scaled_recipe,
        preview_calibration,
        progress=progress,
        cancelled=cancelled,
    )


def _analyze_region_preview(
    region,
    bounds,
    rois,
    include_ids,
    recipe,
    calibration,
    *,
    progress,
    cancelled,
):
    """Analyze a selected source ROI at its independently selected resolution."""
    left, top, right, bottom = bounds
    local_rois = [
        roi.model_copy(
            update={
                "points": [
                    Point(x=point.x - left, y=point.y - top) for point in roi.points
                ]
            }
        )
        for roi in rois
    ]
    progress(0.005, "Preparing selected-ROI preview")
    return _analyze_preview(
        region,
        (right - left, bottom - top),
        local_rois,
        include_ids,
        recipe,
        calibration,
        progress=progress,
        cancelled=cancelled,
    )


def _region_training_labels(
    strokes: list[TrainingStroke],
    class_ids: list[str],
    bounds: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize source-coordinate training strokes into one selected ROI crop."""
    left, top, right, bottom = bounds
    height, width = shape
    scale_x = width / (right - left)
    scale_y = height / (bottom - top)
    labels = np.zeros((height, width), dtype=np.uint16)
    class_values = {class_id: index + 1 for index, class_id in enumerate(class_ids)}
    for stroke in strokes:
        value = 0 if stroke.erase else class_values.get(stroke.class_id)
        if value is None or not stroke.points:
            continue
        coordinates = [
            (
                round((point.x - left) * scale_x),
                round((point.y - top) * scale_y),
            )
            for point in stroke.points
        ]
        radius = max(1, round(stroke.radius_px * (scale_x + scale_y) / 2))
        if len(coordinates) == 1:
            cv2.circle(labels, coordinates[0], radius, value, cv2.FILLED)
        else:
            cv2.line(
                labels,
                coordinates[0],
                coordinates[-1],
                value,
                2 * radius + 1,
            )
    return labels


def _classify_source_region(
    path: str | Path,
    bounds: tuple[int, int, int, int],
    resolution: int,
    training_labels: np.ndarray,
    recipe: RegionClassifierRecipe,
    roi_id: str,
    *,
    progress,
    cancelled,
):
    """Load and classify only one source ROI at its selected working resolution."""
    progress(0.01, "Loading selected region-classification ROI")
    region, actual_bounds = load_region(
        path, bounds, color=True, scale_percent=resolution
    )
    if training_labels.shape != region.shape[:2]:
        training_labels = cv2.resize(
            training_labels,
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    result = classify_regions(
        region,
        training_labels,
        recipe,
        progress=progress,
        cancelled=cancelled,
    )
    return result, actual_bounds, roi_id


def _save_color_plot_outputs(
    figure, destination: str | Path, stem: str
) -> list[Path]:
    """Save the combined plot, every individual graph, and a reloadable Figure pickle."""
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    combined = output / f"{stem}.png"
    figure.savefig(combined, dpi=200, bbox_inches="tight")
    written.append(combined)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    graph_names = ("size_histogram", "size_circularity", "volume_fraction")
    for axis, graph_name in zip(figure.axes, graph_names, strict=False):
        extent = axis.get_tightbbox(renderer).expanded(1.04, 1.08)
        extent = extent.transformed(figure.dpi_scale_trans.inverted())
        target = output / f"{stem}_{graph_name}.png"
        figure.savefig(target, dpi=200, bbox_inches=extent)
        written.append(target)
    pickle_path = output / f"{stem}.figure.pickle"
    with pickle_path.open("wb") as handle:
        pickle.dump(figure, handle, protocol=pickle.HIGHEST_PROTOCOL)
    written.append(pickle_path)
    return written


def _particle_volume_pie_data(
    buckets: list[tuple[str, str, list]], analyzed_pixels: int | None
) -> tuple[list[str], list[float], list[str], str]:
    """Build area-based volume-fraction slices using the analysis domain."""
    labels: list[str] = []
    areas: list[float] = []
    colors: list[str] = []
    for name, color, records in buckets:
        area = float(sum(particle.area_px for particle in records))
        if area > 0:
            labels.append(name)
            areas.append(area)
            colors.append(color)
    visible_area = sum(areas)
    if analyzed_pixels:
        remainder = max(0.0, float(analyzed_pixels) - visible_area)
        if remainder > 0:
            labels.append("Matrix / unshown")
            areas.append(remainder)
            colors.append("#30343b")
        title = "Estimated volume fraction\n(area basis)"
    else:
        title = "Particle area share"
    return labels, areas, colors, title


class MaskLineCommand(QUndoCommand):
    def __init__(self, window: MainWindow, points: list[QPointF], erase: bool) -> None:
        super().__init__("Erase mask" if erase else "Paint mask")
        self.window = window
        self.erase = erase
        radius = window.brush_radius.value()
        coordinates = [(round(point.x()), round(point.y())) for point in points]
        self.coordinates = coordinates
        xs, ys = [p[0] for p in coordinates], [p[1] for p in coordinates]
        self.x1 = max(0, min(xs) - radius - 2)
        self.y1 = max(0, min(ys) - radius - 2)
        self.x2 = min(window.current_mask.shape[1], max(xs) + radius + 3)
        self.y2 = min(window.current_mask.shape[0], max(ys) + radius + 3)
        self.before = window.current_mask[self.y1 : self.y2, self.x1 : self.x2].copy()
        self.after: np.ndarray | None = None

    def _draw(self) -> None:
        target = self.window.current_mask
        value = 0 if self.erase else 1
        radius = self.window.brush_radius.value()
        if len(self.coordinates) == 1:
            cv2.circle(target, self.coordinates[0], radius, value, cv2.FILLED)
        else:
            cv2.line(target, self.coordinates[0], self.coordinates[-1], value, 2 * radius + 1)

    def redo(self) -> None:
        if self.after is None:
            self._draw()
            self.after = self.window.current_mask[self.y1 : self.y2, self.x1 : self.x2].copy()
        else:
            self.window.current_mask[self.y1 : self.y2, self.x1 : self.x2] = self.after
        self.window.schedule_mask_commit()

    def undo(self) -> None:
        self.window.current_mask[self.y1 : self.y2, self.x1 : self.x2] = self.before
        self.window.schedule_mask_commit()


class RegionPaintCommand(QUndoCommand):
    def __init__(
        self, window: MainWindow, points: list[QPointF], value: int, erase: bool
    ) -> None:
        super().__init__("Erase region label" if erase else "Paint region label")
        self.window = window
        self.value = 0 if erase else value
        radius = window.brush_radius.value()
        self.coordinates = [(round(point.x()), round(point.y())) for point in points]
        xs, ys = [point[0] for point in self.coordinates], [point[1] for point in self.coordinates]
        self.x1 = max(0, min(xs) - radius - 2)
        self.y1 = max(0, min(ys) - radius - 2)
        self.x2 = min(window.current_labels.shape[1], max(xs) + radius + 3)
        self.y2 = min(window.current_labels.shape[0], max(ys) + radius + 3)
        self.before = window.current_labels[self.y1 : self.y2, self.x1 : self.x2].copy()
        self.after: np.ndarray | None = None

    def _draw(self) -> None:
        radius = self.window.brush_radius.value()
        if len(self.coordinates) == 1:
            cv2.circle(
                self.window.current_labels,
                self.coordinates[0],
                radius,
                self.value,
                cv2.FILLED,
            )
        else:
            cv2.line(
                self.window.current_labels,
                self.coordinates[0],
                self.coordinates[-1],
                self.value,
                2 * radius + 1,
            )

    def redo(self) -> None:
        if self.after is None:
            self._draw()
            self.after = self.window.current_labels[
                self.y1 : self.y2, self.x1 : self.x2
            ].copy()
        else:
            self.window.current_labels[self.y1 : self.y2, self.x1 : self.x2] = self.after
        self.window._commit_region_edit()

    def undo(self) -> None:
        self.window.current_labels[self.y1 : self.y2, self.x1 : self.x2] = self.before
        self.window._commit_region_edit()


def _slider_control(spinbox: QSpinBox | QDoubleSpinBox) -> tuple[QWidget, QSlider]:
    """Pair a precise spin box with a quickly adjustable horizontal slider."""
    control = QWidget()
    layout = QHBoxLayout(control)
    layout.setContentsMargins(0, 0, 0, 0)
    slider = QSlider(Qt.Orientation.Horizontal)
    if isinstance(spinbox, QDoubleSpinBox):
        scale = 10**spinbox.decimals()
        slider.setRange(round(spinbox.minimum() * scale), round(spinbox.maximum() * scale))
        slider.setSingleStep(max(1, round(spinbox.singleStep() * scale)))
        slider.valueChanged.connect(lambda value: spinbox.setValue(value / scale))
        spinbox.valueChanged.connect(lambda value: slider.setValue(round(value * scale)))
        slider.setValue(round(spinbox.value() * scale))
    else:
        slider.setRange(spinbox.minimum(), spinbox.maximum())
        slider.setSingleStep(spinbox.singleStep())
        slider.setPageStep(max(spinbox.singleStep(), (spinbox.maximum() - spinbox.minimum()) // 20))
        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)
        slider.setValue(spinbox.value())
    layout.addWidget(slider, 1)
    layout.addWidget(spinbox)
    return control, slider


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GST Image — Weld Microstructure Analysis")
        self.resize(1500, 920)
        self.canvas = ImageCanvas()
        self.setCentralWidget(self.canvas)
        self.thread_pool = QThreadPool.globalInstance()
        self.worker: FunctionWorker | None = None
        self.undo_stack = QUndoStack(self)
        self.current_path: Path | None = None
        self.current_project: Path | None = None
        self.preview_bgr: np.ndarray | None = None
        self.display_region_bgr: np.ndarray | None = None
        self.display_region_bounds: tuple[int, int, int, int] | None = None
        self.gray: np.ndarray | None = None
        self.analysis_mask: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.current_labels: np.ndarray | None = None
        self.current_layer_id: str | None = None
        self.training_labels: np.ndarray | None = None
        self.manifest: ProjectManifest | None = None
        self.layer_masks: dict[str, np.ndarray] = {}
        self.particles = []
        self.grouping_preview: ParticleGroupingResult | None = None
        self.grouping_preview_definition: ParticleGrouping | None = None
        self._pending_grouping_preview: ParticleGrouping | None = None
        self._synchronizing_class_selection = False
        self.selected_labels: set[int] = set()
        self.run_scope_ids: list[str] = []
        self.run_recipe: SegmentationRecipe | None = None
        self.confirmed_preview_recipe: SegmentationRecipe | None = None
        self.confirmed_preview_scope_ids: list[str] = []
        self._pending_preview_recipe: SegmentationRecipe | None = None
        self._pending_preview_scope_ids: list[str] = []
        self._pending_preview_full_resolution = False
        self._pending_preview_resolution: int | None = None
        self.dirty = False
        self._build_actions()
        self._build_analysis_dock()
        self._build_results_dock()
        self._connect_canvas()
        self.mask_commit_timer = QTimer(self)
        self.mask_commit_timer.setSingleShot(True)
        self.mask_commit_timer.setInterval(250)
        self.mask_commit_timer.timeout.connect(self._commit_mask_edits)
        self.grouping_preview_timer = QTimer(self)
        self.grouping_preview_timer.setSingleShot(True)
        self.grouping_preview_timer.setInterval(100)
        self.grouping_preview_timer.timeout.connect(self._apply_pending_grouping_preview)
        self.statusBar().showMessage("Open a micrograph or project to begin")

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, shortcut, slot in (
            ("Open image…", "Ctrl+O", self.open_image),
            ("Open project…", "Ctrl+Shift+O", self.open_project),
            ("Save project…", "Ctrl+S", self.save_project_dialog),
            ("Export…", "Ctrl+E", self.export_dialog),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        portable_action = QAction("Save portable project as…", self)
        portable_action.triggered.connect(self.save_portable_project)
        file_menu.addAction(portable_action)
        relink_action = QAction("Relink source as new revision…", self)
        relink_action.triggered.connect(self.relink_source_dialog)
        file_menu.addAction(relink_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_stack.createUndoAction(self, "Undo"))
        edit_menu.addAction(self.undo_stack.createRedoAction(self, "Redo"))
        edit_menu.addSeparator()
        self.decrease_brush_action = QAction("Decrease brush radius", self)
        self.decrease_brush_action.setShortcut(QKeySequence("["))
        self.decrease_brush_action.triggered.connect(
            lambda: self._adjust_brush_radius(-1)
        )
        self.decrease_brush_action.setEnabled(False)
        edit_menu.addAction(self.decrease_brush_action)
        self.increase_brush_action = QAction("Increase brush radius", self)
        self.increase_brush_action.setShortcut(QKeySequence("]"))
        self.increase_brush_action.triggered.connect(
            lambda: self._adjust_brush_radius(1)
        )
        self.increase_brush_action.setEnabled(False)
        edit_menu.addAction(self.increase_brush_action)

        analysis_menu = self.menuBar().addMenu("&Analysis")
        self.preview_action = QAction("Preview", self)
        self.preview_action.setShortcut(QKeySequence("Shift+A"))
        self.preview_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.preview_action.triggered.connect(self.run_preview_segmentation)
        analysis_menu.addAction(self.preview_action)

        toolbar = QToolBar("Tools", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        group = QActionGroup(self)
        group.setExclusive(True)
        tools = [
            ("Pan", "pan"),
            ("Select", "select"),
            ("Calibrate", "calibrate"),
            ("Measure", "measure"),
            ("Eyedropper", "eyedropper"),
            ("Include box", "include"),
            ("Analysis box", "analysis_box"),
            ("Exclude box", "exclude"),
            ("Polygon", "polygon"),
            ("Class seed", "seed"),
            ("Seed eraser", "seed_eraser"),
            ("Mask brush", "mask_brush"),
            ("Mask eraser", "mask_eraser"),
            ("Split", "split"),
        ]
        self.tool_actions: dict[str, QAction] = {}
        for label, name in tools:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, value=name: self._activate_tool(value)
            )
            toolbar.addAction(action)
            group.addAction(action)
            self.tool_actions[name] = action
        for name, shortcut in (("mask_brush", "O"), ("mask_eraser", "P")):
            self.tool_actions[name].setShortcut(QKeySequence(shortcut))
            self.tool_actions[name].setShortcutContext(
                Qt.ShortcutContext.ApplicationShortcut
            )
        self.tool_actions["pan"].setChecked(True)
        toolbar.addSeparator()
        merge = QAction("Merge selected", self)
        merge.triggered.connect(self.merge_selected)
        toolbar.addAction(merge)

    def _build_analysis_dock(self) -> None:
        dock = QDockWidget("Analysis", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()
        self.analysis_form = form
        self.binary_class = QComboBox()
        self.channel = QComboBox()
        for label, value in (
            ("Grayscale", "gray"),
            ("Blue", "blue"),
            ("Green", "green"),
            ("Red", "red"),
            ("Lab lightness", "lab_l"),
            ("Lab a", "lab_a"),
            ("Lab b", "lab_b"),
        ):
            self.channel.addItem(label, value)
        self.segmentation_method = QComboBox()
        self.segmentation_method.addItem("Threshold particles", SegmentationMethod.THRESHOLD)
        self.segmentation_method.addItem(
            "Morphological watershed", SegmentationMethod.MORPHOLOGICAL_WATERSHED
        )
        self.method = QComboBox()
        for method in ThresholdMethod:
            self.method.addItem(method.value.replace("_", " ").title(), method)
        self.polarity = QComboBox()
        for polarity in ParticlePolarity:
            self.polarity.addItem(polarity.value.title(), polarity)
        self.illumination = QCheckBox("Rolling-ball correction")
        self.illumination.setChecked(True)
        self.split_touching = QCheckBox("Watershed split touching particles")
        self.split_touching.setChecked(True)
        self.selected_roi_only = QCheckBox("Analyze selected ROI/box only")
        self.overview_resolution = QSpinBox()
        self.overview_resolution.setRange(0, 100)
        self.overview_resolution.setSuffix("%")
        self.overview_resolution.setValue(25)
        self.overview_resolution.setToolTip(
            "Percentage of native width and height; 100% loads the full overview."
        )
        self.roi_resolution = QSpinBox()
        self.roi_resolution.setRange(0, 100)
        self.roi_resolution.setSuffix("%")
        self.roi_resolution.setValue(100)
        self.roi_resolution.setToolTip(
            "Independent resolution for the selected Include/Analysis ROI."
        )
        self.preview_resolution = QComboBox()
        self.preview_resolution.addItem("Fast overview", "overview")
        self.preview_resolution.addItem("Selected ROI at full resolution", "roi_full")
        self.eyedropper_target = QComboBox()
        self.eyedropper_target.addItem("Manual upper threshold", "threshold_high")
        self.eyedropper_target.addItem("Manual lower threshold", "threshold_low")
        self.eyedropper_target.addItem("Segmentation class color", "binary_color")
        self.eyedropper_target.addItem("Region class color", "region_color")
        self.window_size = QSpinBox()
        self.window_size.setRange(3, 2001)
        self.window_size.setSingleStep(2)
        self.window_size.setValue(101)
        self.sauvola_k = QDoubleSpinBox()
        self.sauvola_k.setRange(-1, 1)
        self.sauvola_k.setDecimals(3)
        self.sauvola_k.setSingleStep(0.01)
        self.sauvola_k.setValue(0.2)
        self.gaussian_block = QSpinBox()
        self.gaussian_block.setRange(3, 2001)
        self.gaussian_block.setSingleStep(2)
        self.gaussian_block.setValue(101)
        self.gaussian_c = QDoubleSpinBox()
        self.gaussian_c.setRange(-255, 255)
        self.gaussian_c.setDecimals(2)
        self.gaussian_c.setSingleStep(0.5)
        self.gaussian_c.setValue(2.0)
        self.manual_threshold_low = QSpinBox()
        self.manual_threshold_low.setRange(0, 255)
        self.manual_threshold_low.setValue(0)
        self.manual_threshold_high = QSpinBox()
        self.manual_threshold_high.setRange(0, 255)
        self.manual_threshold_high.setValue(127)
        self.manual_threshold_low.valueChanged.connect(self._manual_threshold_low_changed)
        self.manual_threshold_high.valueChanged.connect(self._manual_threshold_high_changed)
        self.gaussian_blur_sigma = QDoubleSpinBox()
        self.gaussian_blur_sigma.setRange(0, 20)
        self.gaussian_blur_sigma.setDecimals(2)
        self.gaussian_blur_sigma.setSingleStep(0.1)
        self.gaussian_blur_sigma.setValue(0.8)
        self.fill_holes = QCheckBox("Flood-fill enclosed holes")
        self.fill_holes.setToolTip(
            "Fill background regions that cannot be reached from the analysis-domain border."
        )
        self.ball_radius = QSpinBox()
        self.ball_radius.setRange(3, 10001)
        self.ball_radius.setValue(151)
        self.open_radius = QSpinBox()
        self.open_radius.setRange(0, 500)
        self.open_radius.setValue(1)
        self.close_radius = QSpinBox()
        self.close_radius.setRange(0, 500)
        self.close_radius.setValue(1)
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 10_000_000)
        self.min_area.setValue(25)
        self.max_area = QSpinBox()
        self.max_area.setRange(0, 100_000_000)
        self.max_area.setSpecialValueText("No maximum")
        self.tile_size = QSpinBox()
        self.tile_size.setRange(256, 8192)
        self.tile_size.setSingleStep(256)
        self.tile_size.setValue(2048)
        self.watershed_distance = QSpinBox()
        self.watershed_distance.setRange(1, 1000)
        self.watershed_distance.setValue(7)
        self.morphological_input = QComboBox()
        self.morphological_input.addItem("Object image", MorphologicalInput.OBJECT)
        self.morphological_input.addItem("Border image", MorphologicalInput.BORDER)
        self.morphological_gradient = QComboBox()
        self.morphological_gradient.addItem(
            "Morphological", MorphologicalGradient.MORPHOLOGICAL
        )
        self.morphological_gradient.addItem("Internal", MorphologicalGradient.INTERNAL)
        self.morphological_gradient.addItem("External", MorphologicalGradient.EXTERNAL)
        self.morphological_gradient_radius = QSpinBox()
        self.morphological_gradient_radius.setRange(1, 100)
        self.morphological_gradient_radius.setValue(1)
        self.morphological_tolerance = QSpinBox()
        self.morphological_tolerance.setRange(1, 255)
        self.morphological_tolerance.setValue(10)
        self.morphological_tolerance.setToolTip(
            "Extended-minima depth; higher values merge shallow basins into fewer segments."
        )
        self.morphological_connectivity = QComboBox()
        self.morphological_connectivity.addItem("4-connected", 4)
        self.morphological_connectivity.addItem("8-connected", 8)
        self.morphological_calculate_dams = QCheckBox("Calculate watershed dams")
        self.morphological_calculate_dams.setChecked(True)
        self.brush_radius = QSpinBox()
        self.brush_radius.setRange(1, 500)
        self.brush_radius.setValue(12)
        self.threshold_defaults_button = QPushButton("Restore threshold defaults")
        self.threshold_defaults_button.clicked.connect(self.restore_threshold_defaults)
        self.morphological_defaults_button = QPushButton(
            "Restore morphological defaults"
        )
        self.morphological_defaults_button.clicked.connect(
            self.restore_morphological_defaults
        )
        self.preblur_defaults_button = QPushButton("Restore pre-blur default")
        self.preblur_defaults_button.clicked.connect(self.restore_preblur_defaults)
        self.rolling_ball_defaults_button = QPushButton(
            "Restore rolling-ball defaults"
        )
        self.rolling_ball_defaults_button.clicked.connect(
            self.restore_rolling_ball_defaults
        )
        self.open_close_defaults_button = QPushButton("Restore open/close defaults")
        self.open_close_defaults_button.clicked.connect(
            self.restore_open_close_defaults
        )
        self.flood_fill_defaults_button = QPushButton("Restore flood-fill default")
        self.flood_fill_defaults_button.clicked.connect(
            self.restore_flood_fill_defaults
        )
        self.particle_filter_defaults_button = QPushButton(
            "Restore particle-filter defaults"
        )
        self.particle_filter_defaults_button.clicked.connect(
            self.restore_particle_filter_defaults
        )
        self.watershed_defaults_button = QPushButton("Restore watershed defaults")
        self.watershed_defaults_button.clicked.connect(
            self.restore_watershed_defaults
        )
        self.tile_defaults_button = QPushButton("Restore tile-size default")
        self.tile_defaults_button.clicked.connect(self.restore_tile_defaults)
        self.brush_defaults_button = QPushButton("Restore brush default")
        self.brush_defaults_button.clicked.connect(self.restore_brush_defaults)
        form.addRow("Segmentation class", self.binary_class)
        form.addRow("Analysis channel", self.channel)
        form.addRow("Segmentation method", self.segmentation_method)
        form.addRow("Threshold method", self.method)
        form.addRow("Particle polarity", self.polarity)
        overview_control, self.overview_resolution_slider = _slider_control(
            self.overview_resolution
        )
        roi_resolution_control, self.roi_resolution_slider = _slider_control(
            self.roi_resolution
        )
        form.addRow("Overview resolution", overview_control)
        form.addRow("Selected ROI resolution", roi_resolution_control)
        form.addRow("Preview resolution", self.preview_resolution)
        form.addRow("Eyedropper sets", self.eyedropper_target)
        self.sauvola_window_control, self.window_size_slider = _slider_control(
            self.window_size
        )
        self.sauvola_k_control, self.sauvola_k_slider = _slider_control(self.sauvola_k)
        self.gaussian_block_control, self.gaussian_block_slider = _slider_control(
            self.gaussian_block
        )
        self.gaussian_c_control, self.gaussian_c_slider = _slider_control(self.gaussian_c)
        self.manual_threshold_low_control, self.manual_threshold_low_slider = _slider_control(
            self.manual_threshold_low
        )
        self.manual_threshold_high_control, self.manual_threshold_high_slider = _slider_control(
            self.manual_threshold_high
        )
        self.gaussian_blur_control, self.gaussian_blur_sigma_slider = _slider_control(
            self.gaussian_blur_sigma
        )
        self.ball_radius_control, self.ball_radius_slider = _slider_control(self.ball_radius)
        open_control, self.open_radius_slider = _slider_control(self.open_radius)
        close_control, self.close_radius_slider = _slider_control(self.close_radius)
        min_area_control, self.min_area_slider = _slider_control(self.min_area)
        max_area_control, self.max_area_slider = _slider_control(self.max_area)
        self.watershed_distance_control, self.watershed_distance_slider = _slider_control(
            self.watershed_distance
        )
        self.morphological_gradient_control, self.morphological_gradient_slider = (
            _slider_control(self.morphological_gradient_radius)
        )
        self.morphological_tolerance_control, self.morphological_tolerance_slider = (
            _slider_control(self.morphological_tolerance)
        )
        tile_control, self.tile_size_slider = _slider_control(self.tile_size)
        brush_control, self.brush_radius_slider = _slider_control(self.brush_radius)
        form.addRow("Sauvola window (px)", self.sauvola_window_control)
        form.addRow("Sauvola k", self.sauvola_k_control)
        form.addRow("Adaptive block size (px)", self.gaussian_block_control)
        form.addRow("Adaptive constant C", self.gaussian_c_control)
        form.addRow("Manual lower threshold", self.manual_threshold_low_control)
        form.addRow("Manual upper threshold", self.manual_threshold_high_control)
        form.addRow(self.threshold_defaults_button)
        form.addRow("Morphological input", self.morphological_input)
        form.addRow("Gradient type", self.morphological_gradient)
        form.addRow("Morphological gradient radius (px)", self.morphological_gradient_control)
        form.addRow("Morphological tolerance", self.morphological_tolerance_control)
        form.addRow("Morphological connectivity", self.morphological_connectivity)
        form.addRow(self.morphological_calculate_dams)
        form.addRow(self.morphological_defaults_button)
        form.addRow("Gaussian pre-blur sigma (px)", self.gaussian_blur_control)
        form.addRow(self.preblur_defaults_button)
        form.addRow(self.illumination)
        form.addRow("Background radius (px)", self.ball_radius_control)
        form.addRow(self.rolling_ball_defaults_button)
        form.addRow("Opening radius (px)", open_control)
        form.addRow("Closing radius (px)", close_control)
        form.addRow(self.open_close_defaults_button)
        form.addRow(self.fill_holes)
        form.addRow(self.flood_fill_defaults_button)
        form.addRow("Minimum area (px²)", min_area_control)
        form.addRow("Maximum area (px²)", max_area_control)
        form.addRow(self.particle_filter_defaults_button)
        form.addRow(self.split_touching)
        form.addRow("Watershed minimum distance (px)", self.watershed_distance_control)
        form.addRow(self.watershed_defaults_button)
        form.addRow("Tile size (px)", tile_control)
        form.addRow(self.tile_defaults_button)
        form.addRow("Brush radius (px)", brush_control)
        form.addRow(self.brush_defaults_button)
        layout.addLayout(form)
        layout.addWidget(self.selected_roi_only)
        display_row = QHBoxLayout()
        self.full_resolution_button = QPushButton("Show selected ROI")
        self.full_resolution_button.clicked.connect(self.show_full_resolution_roi)
        self.overview_button = QPushButton("Show overview")
        self.overview_button.clicked.connect(self.show_overview)
        display_row.addWidget(self.full_resolution_button)
        display_row.addWidget(self.overview_button)
        layout.addLayout(display_row)
        row = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.run_preview_segmentation)
        self.run_button = QPushButton("Run particles — full resolution")
        self.run_button.clicked.connect(self.run_segmentation)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        row.addWidget(self.preview_button)
        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        self.preview_status = QLabel("Preview parameters not confirmed")
        self.preview_status.setWordWrap(True)
        layout.addWidget(self.preview_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        layout.addWidget(self.progress)
        for control in (
            self.binary_class,
            self.channel,
            self.segmentation_method,
            self.method,
            self.polarity,
            self.overview_resolution,
            self.roi_resolution,
            self.preview_resolution,
            self.illumination,
            self.split_touching,
            self.selected_roi_only,
            self.window_size,
            self.sauvola_k,
            self.gaussian_block,
            self.gaussian_c,
            self.manual_threshold_low,
            self.manual_threshold_high,
            self.gaussian_blur_sigma,
            self.fill_holes,
            self.ball_radius,
            self.open_radius,
            self.close_radius,
            self.min_area,
            self.max_area,
            self.watershed_distance,
            self.morphological_input,
            self.morphological_gradient,
            self.morphological_gradient_radius,
            self.morphological_tolerance,
            self.morphological_connectivity,
            self.morphological_calculate_dams,
            self.tile_size,
        ):
            signal = (
                control.toggled
                if isinstance(control, QCheckBox)
                else control.currentIndexChanged
                if isinstance(control, QComboBox)
                else control.valueChanged
            )
            signal.connect(self._invalidate_preview_confirmation)
        self.method.currentIndexChanged.connect(self._update_analysis_parameter_visibility)
        self.segmentation_method.currentIndexChanged.connect(
            self._update_analysis_parameter_visibility
        )
        self.morphological_input.currentIndexChanged.connect(
            self._update_analysis_parameter_visibility
        )
        self.illumination.toggled.connect(self._update_analysis_parameter_visibility)
        self.split_touching.toggled.connect(self._update_analysis_parameter_visibility)
        self._update_analysis_parameter_visibility()
        layout.addWidget(QLabel("Assisted region classes"))
        self.seed_class = QComboBox()
        layout.addWidget(self.seed_class)
        self.binary_class.currentIndexChanged.connect(
            self._segmentation_class_changed
        )
        self.seed_class.currentIndexChanged.connect(self._assisted_class_changed)
        class_buttons = QHBoxLayout()
        add_class = QPushButton("Add class")
        add_class.clicked.connect(self.add_class)
        rename_class = QPushButton("Rename class")
        rename_class.clicked.connect(self.rename_class)
        class_buttons.addWidget(add_class)
        class_buttons.addWidget(rename_class)
        layout.addLayout(class_buttons)
        self.train_regions_button = QPushButton("Train / update region classifier")
        self.train_regions_button.setToolTip(
            "Classifies the selected Analysis box at Selected ROI resolution; "
            "the crop must be no larger than 4.5 megapixels."
        )
        self.train_regions_button.clicked.connect(self.train_regions)
        layout.addWidget(self.train_regions_button)
        self.run_model_button = QPushButton("Run reviewed model pack…")
        self.run_model_button.setToolTip(
            "Runs a local ONNX model pack at native resolution. Results remain pending expert review."
        )
        self.run_model_button.clicked.connect(self.run_model_inference_dialog)
        layout.addWidget(self.run_model_button)
        self.run_cellpose_button = QPushButton("Run Cellpose-SAM v2…")
        self.run_cellpose_button.setToolTip(
            "Runs local stock Cellpose-SAM v2 at native resolution. The first weight download is explicit."
        )
        self.run_cellpose_button.clicked.connect(self.run_cellpose_inference_dialog)
        layout.addWidget(self.run_cellpose_button)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_results_dock(self) -> None:
        dock = QDockWidget("Results and layers", self)
        tabs = QTabWidget()
        self.summary = QLabel("No analysis yet")
        self.summary.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary.setWordWrap(True)
        tabs.addTab(self.summary, "Summary")
        self.particle_table = QTableWidget(0, 7)
        self.particle_table.setHorizontalHeaderLabels(
            ["Label", "Size", "Circularity", "Area px²", "Solidity", "Border", "Group"]
        )
        self.particle_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tabs.addTab(self.particle_table, "Particles")
        layer_panel = QWidget()
        layer_layout = QVBoxLayout(layer_panel)
        self.layers_list = QListWidget()
        self.layers_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.layers_list.itemClicked.connect(self.show_selected_layer)
        self.layers_list.itemChanged.connect(self._layer_visibility_changed)
        layer_layout.addWidget(self.layers_list)
        self.delete_layers_button = QPushButton("Delete selected")
        self.delete_layers_button.clicked.connect(self.delete_selected_layers)
        layer_layout.addWidget(self.delete_layers_button)
        self.confirm_model_button = QPushButton("Confirm selected model result…")
        self.confirm_model_button.clicked.connect(self.confirm_selected_model_result)
        layer_layout.addWidget(self.confirm_model_button)
        self.copy_roi_mask_button = QPushButton("Copy ROI mask image")
        self.copy_roi_mask_button.clicked.connect(self.copy_current_roi_mask)
        layer_layout.addWidget(self.copy_roi_mask_button)
        self.layer_opacity = QDoubleSpinBox()
        self.layer_opacity.setRange(0, 1)
        self.layer_opacity.setSingleStep(0.05)
        self.layer_opacity.setValue(0.45)
        self.layer_opacity.valueChanged.connect(self._layer_opacity_changed)
        layer_layout.addWidget(QLabel("Selected layer opacity"))
        layer_layout.addWidget(self.layer_opacity)
        tabs.addTab(layer_panel, "Layers")
        roi_panel = QWidget()
        roi_layout = QVBoxLayout(roi_panel)
        self.rois_list = QListWidget()
        self.rois_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.rois_list.itemClicked.connect(self._roi_selected)
        self.rois_list.itemSelectionChanged.connect(self._invalidate_preview_confirmation)
        roi_layout.addWidget(self.rois_list)
        self.delete_rois_button = QPushButton("Delete selected")
        self.delete_rois_button.clicked.connect(self.delete_selected_rois)
        roi_layout.addWidget(self.delete_rois_button)
        tabs.addTab(roi_panel, "ROIs")
        measurement_panel = QWidget()
        measurement_layout = QVBoxLayout(measurement_panel)
        self.measurements_list = QListWidget()
        self.measurements_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        measurement_layout.addWidget(self.measurements_list)
        self.delete_measurements_button = QPushButton("Delete selected")
        self.delete_measurements_button.clicked.connect(self.delete_selected_measurements)
        measurement_layout.addWidget(self.delete_measurements_button)
        tabs.addTab(measurement_panel, "Measurements")
        self.grouping_panel = ParticleGroupingPanel()
        self.grouping_panel.draftChanged.connect(self._queue_grouping_preview)
        self.grouping_panel.saveRequested.connect(self._save_particle_grouping)
        self.grouping_panel.deleteRequested.connect(self._delete_particle_grouping)
        self.grouping_panel.plotRequested.connect(self.plot_particles)
        self.grouping_panel.saveImageRequested.connect(
            self.save_particle_grouping_overlay
        )
        # Compatibility alias for callers that previously accessed the simple group table.
        self.groups_table = self.grouping_panel.groups_table
        group_scroll = QScrollArea()
        group_scroll.setWidgetResizable(True)
        group_scroll.setWidget(self.grouping_panel)
        tabs.addTab(group_scroll, "Particle groups")
        dock.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _connect_canvas(self) -> None:
        self.canvas.line_finished.connect(self._line_finished)
        self.canvas.rectangle_finished.connect(self._rectangle_finished)
        self.canvas.polygon_finished.connect(self._polygon_finished)
        self.canvas.brush_stroke.connect(self._brush_stroke)
        self.canvas.brush_radius_adjust_requested.connect(self._adjust_brush_radius)
        self.canvas.scene_clicked.connect(self._select_particle)
        self.canvas.coordinates_changed.connect(
            lambda point: self.statusBar().showMessage(f"x={point.x():.1f}, y={point.y():.1f}")
        )
        self.brush_radius.valueChanged.connect(self.canvas.set_brush_radius)
        self.canvas.set_brush_radius(self.brush_radius.value())

    def _activate_tool(self, tool: str) -> None:
        """Activate a canvas tool and keep brush-only controls/context in sync."""
        self.canvas.set_tool(tool)
        brush_active = tool in self.canvas.BRUSH_TOOLS
        self.decrease_brush_action.setEnabled(brush_active)
        self.increase_brush_action.setEnabled(brush_active)
        if tool in {"seed", "seed_eraser"}:
            self._render_training_overlay()
        elif self.manifest is not None:
            self._render_visible_layers()
        if tool in {"mask_brush", "mask_eraser"}:
            layer = self._current_editable_layer()
            if layer is None:
                self.statusBar().showMessage(
                    "No editable mask is selected. Run particles or region "
                    "classification, then select its layer.",
                    6000,
                )
            else:
                self.statusBar().showMessage(
                    f"Editing {layer.name}; brush radius: {self.brush_radius.value()} px",
                    4000,
                )

    def _adjust_brush_radius(self, delta: int) -> None:
        if self.canvas.current_tool not in self.canvas.BRUSH_TOOLS:
            return
        previous = self.brush_radius.value()
        self.brush_radius.setValue(previous + delta)
        current = self.brush_radius.value()
        if current != previous:
            self.statusBar().showMessage(f"Brush radius: {current} px", 2000)

    def _segmentation_class_changed(self, _index: int) -> None:
        """Keep particle-result and assisted-region class selections aligned."""
        self._synchronize_class_selection(self.binary_class, self.seed_class)

    def _assisted_class_changed(self, _index: int) -> None:
        """Use the selected assisted class for subsequent particle results."""
        self._synchronize_class_selection(self.seed_class, self.binary_class)

    def _synchronize_class_selection(self, source: QComboBox, target: QComboBox) -> None:
        if self._synchronizing_class_selection:
            return
        class_id = source.currentData()
        target_index = target.findData(class_id)
        if class_id is None or target_index < 0 or target_index == target.currentIndex():
            return
        self._synchronizing_class_selection = True
        try:
            target.setCurrentIndex(target_index)
        finally:
            self._synchronizing_class_selection = False

    def confirm_selected_model_result(self) -> None:
        if self.manifest is None or self.current_layer_id is None:
            return
        layer = next(
            (item for item in self.manifest.layers if item.id == self.current_layer_id), None
        )
        run = next(
            (
                item
                for item in [
                    *self.manifest.model_inference_runs,
                    *self.manifest.cellpose_inference_runs,
                ]
                if layer is not None and item.id == layer.source_run_id
            ),
            None,
        )
        if run is None:
            QMessageBox.information(
                self, "Confirm model result", "Select a layer produced by a pending model run first."
            )
            return
        if run.review_status == "confirmed":
            QMessageBox.information(self, "Confirm model result", "This model result is already confirmed.")
            return
        reviewer, accepted = QInputDialog.getText(
            self, "Confirm model result", "Reviewer name or identifier:"
        )
        if not accepted or not reviewer.strip():
            return
        run.review_status = "confirmed"
        run.reviewed_at = datetime.now(UTC)
        run.reviewer = reviewer.strip()
        for item in self.manifest.layers:
            if item.source_run_id == run.id:
                item.review_status = "confirmed"
        self._set_dirty(True)
        self._refresh_all()
        self.statusBar().showMessage(f"Confirmed model result reviewed by {run.reviewer}", 5000)

    def _current_editable_layer(self) -> SegmentationLayer | None:
        if self.manifest is None or self.current_layer_id is None:
            return None
        layer = next(
            (
                item
                for item in self.manifest.layers
                if item.id == self.current_layer_id and item.kind != "domain"
            ),
            None,
        )
        if layer is None:
            return None
        if layer.kind == "multiclass":
            return layer if self.current_labels is not None else None
        return layer if self.current_mask is not None else None

    def _set_dirty(self, dirty: bool = True) -> None:
        self.dirty = dirty
        suffix = " *" if dirty else ""
        self.setWindowTitle(f"GST Image — Weld Microstructure Analysis{suffix}")

    def _invalidate_preview_confirmation(self, *_args) -> None:
        self.confirmed_preview_recipe = None
        self.confirmed_preview_scope_ids = []
        if hasattr(self, "preview_status"):
            self.preview_status.setText("Preview parameters not confirmed")

    def _manual_threshold_low_changed(self, value: int) -> None:
        """Keep the inclusive manual threshold endpoints from crossing."""
        if value > self.manual_threshold_high.value():
            self.manual_threshold_high.setValue(value)

    def _manual_threshold_high_changed(self, value: int) -> None:
        """Keep the inclusive manual threshold endpoints from crossing."""
        if value < self.manual_threshold_low.value():
            self.manual_threshold_low.setValue(value)

    def _set_analysis_row_visible(self, field: QWidget, visible: bool) -> None:
        field.setVisible(visible)
        label = self.analysis_form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    def _update_analysis_parameter_visibility(self, *_args) -> None:
        """Show only controls used by the active thresholding configuration."""
        method = ThresholdMethod(self.method.currentData())
        method_label = method.value.replace("_", " ").title()
        self.threshold_defaults_button.setText(f"Restore {method_label} defaults")
        self.threshold_defaults_button.setEnabled(method != ThresholdMethod.OTSU)
        self.threshold_defaults_button.setToolTip(
            "Otsu calculates its threshold automatically and has no method-specific defaults."
            if method == ThresholdMethod.OTSU
            else "Restore only the parameters used by the selected threshold method."
        )
        self._set_analysis_row_visible(
            self.sauvola_window_control, method == ThresholdMethod.SAUVOLA
        )
        self._set_analysis_row_visible(
            self.sauvola_k_control, method == ThresholdMethod.SAUVOLA
        )
        self._set_analysis_row_visible(
            self.gaussian_block_control,
            method == ThresholdMethod.ADAPTIVE_GAUSSIAN,
        )
        self._set_analysis_row_visible(
            self.gaussian_c_control,
            method == ThresholdMethod.ADAPTIVE_GAUSSIAN,
        )
        self._set_analysis_row_visible(
            self.manual_threshold_low_control, method == ThresholdMethod.MANUAL
        )
        self._set_analysis_row_visible(
            self.manual_threshold_high_control, method == ThresholdMethod.MANUAL
        )
        morphological = (
            self.segmentation_method.currentData()
            == SegmentationMethod.MORPHOLOGICAL_WATERSHED
        )
        self._set_analysis_row_visible(self.morphological_input, morphological)
        self._set_analysis_row_visible(
            self.morphological_gradient,
            morphological and self.morphological_input.currentData() == MorphologicalInput.OBJECT,
        )
        self._set_analysis_row_visible(
            self.morphological_gradient_control,
            morphological and self.morphological_input.currentData() == MorphologicalInput.OBJECT,
        )
        self._set_analysis_row_visible(self.morphological_tolerance_control, morphological)
        self._set_analysis_row_visible(self.morphological_connectivity, morphological)
        self.morphological_calculate_dams.setVisible(morphological)
        self.morphological_defaults_button.setVisible(morphological)
        self._set_analysis_row_visible(
            self.ball_radius_control, self.illumination.isChecked()
        )
        self.split_touching.setVisible(not morphological)
        self.watershed_defaults_button.setVisible(not morphological)
        self._set_analysis_row_visible(
            self.watershed_distance_control,
            not morphological and self.split_touching.isChecked(),
        )

    def _show_restored(self, section: str) -> None:
        self._invalidate_preview_confirmation()
        self.statusBar().showMessage(f"{section} restored to defaults", 3000)

    def restore_threshold_defaults(self) -> None:
        """Restore only parameters belonging to the selected threshold method."""
        defaults = SegmentationRecipe()
        method = ThresholdMethod(self.method.currentData())
        if method == ThresholdMethod.SAUVOLA:
            self.window_size.setValue(defaults.sauvola_window_px)
            self.sauvola_k.setValue(defaults.sauvola_k)
        elif method == ThresholdMethod.ADAPTIVE_GAUSSIAN:
            self.gaussian_block.setValue(defaults.gaussian_block_px)
            self.gaussian_c.setValue(defaults.gaussian_c)
        elif method == ThresholdMethod.MANUAL:
            self.manual_threshold_low.setValue(defaults.manual_threshold_low)
            self.manual_threshold_high.setValue(defaults.manual_threshold_high)
        else:
            self.statusBar().showMessage(
                "Otsu has no method-specific parameters to restore", 3000
            )
            return
        self._show_restored(method.value.replace("_", " ").title())

    def restore_morphological_defaults(self) -> None:
        defaults = SegmentationRecipe()
        self.morphological_input.setCurrentIndex(
            self.morphological_input.findData(defaults.morphological_input)
        )
        self.morphological_gradient.setCurrentIndex(
            self.morphological_gradient.findData(defaults.morphological_gradient)
        )
        self.morphological_gradient_radius.setValue(
            defaults.morphological_gradient_radius_px
        )
        self.morphological_tolerance.setValue(defaults.morphological_tolerance)
        self.morphological_connectivity.setCurrentIndex(
            self.morphological_connectivity.findData(
                defaults.morphological_connectivity
            )
        )
        self.morphological_calculate_dams.setChecked(
            defaults.morphological_calculate_dams
        )
        self._show_restored("Morphological parameters")

    def restore_preblur_defaults(self) -> None:
        self.gaussian_blur_sigma.setValue(SegmentationRecipe().gaussian_blur_sigma)
        self._show_restored("Gaussian pre-blur")

    def restore_rolling_ball_defaults(self) -> None:
        defaults = SegmentationRecipe()
        self.illumination.setChecked(defaults.illumination_correction)
        self.ball_radius.setValue(defaults.rolling_ball_radius_px)
        self._show_restored("Rolling-ball correction")

    def restore_open_close_defaults(self) -> None:
        defaults = SegmentationRecipe()
        self.open_radius.setValue(defaults.open_radius_px)
        self.close_radius.setValue(defaults.close_radius_px)
        self._show_restored("Opening and closing")

    def restore_flood_fill_defaults(self) -> None:
        self.fill_holes.setChecked(SegmentationRecipe().fill_holes)
        self._show_restored("Flood-fill")

    def restore_particle_filter_defaults(self) -> None:
        defaults = SegmentationRecipe()
        self.min_area.setValue(defaults.min_particle_area_px)
        self.max_area.setValue(defaults.max_particle_area_px or 0)
        self._show_restored("Particle filters")

    def restore_watershed_defaults(self) -> None:
        defaults = SegmentationRecipe()
        self.split_touching.setChecked(defaults.split_touching)
        self.watershed_distance.setValue(defaults.watershed_min_distance_px)
        self._show_restored("Touching-particle watershed")

    def restore_tile_defaults(self) -> None:
        self.tile_size.setValue(SegmentationRecipe().tile_size_px)
        self._show_restored("Tile size")

    def restore_brush_defaults(self) -> None:
        self.brush_radius.setValue(12)
        self._show_restored("Brush radius")

    def _selected_full_resolution_roi(self) -> ROI | None:
        if self.manifest is None:
            return None
        selected_ids = {
            item.data(Qt.ItemDataRole.UserRole) for item in self.rois_list.selectedItems()
        }
        if len(selected_ids) != 1:
            return None
        candidates = [
            roi
            for roi in self.manifest.rois
            if roi.id in selected_ids
            and roi.kind in {ROIKind.INCLUDE, ROIKind.ANALYSIS_BOX}
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _selected_analysis_box(self) -> ROI | None:
        """Return the one selected Analysis box used for region classification."""
        roi = self._selected_full_resolution_roi()
        return roi if roi is not None and roi.kind == ROIKind.ANALYSIS_BOX else None

    def _roi_bounds(self, roi: ROI) -> tuple[int, int, int, int]:
        return roi_bounds((self.manifest.image_height, self.manifest.image_width), roi)

    def _set_canvas_image(
        self,
        image: np.ndarray,
        source_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        if self.manifest is None:
            return
        full_size = (self.manifest.image_width, self.manifest.image_height)
        self.canvas.set_image(image, full_size, source_rect)
        self.canvas.set_roi_outlines([self._roi_polygon(roi) for roi in self.manifest.rois])
        self._refresh_measurements()
        self._render_visible_layers()

    def show_overview(self) -> bool:
        if self.current_path is not None and self.current_path.exists():
            try:
                preview, _ = load_scaled_image(
                    self.current_path,
                    self.overview_resolution.value(),
                    color=True,
                )
            except Exception as error:
                QMessageBox.critical(self, "Overview failed", str(error))
                return False
            self.preview_bgr = preview
            self.training_labels = np.zeros(preview.shape[:2], dtype=np.uint16)
            self._restore_training_strokes()
        if self.preview_bgr is None:
            return False
        self.display_region_bgr = None
        self.display_region_bounds = None
        self._set_canvas_image(self.preview_bgr)
        if self.canvas.current_tool in {"seed", "seed_eraser"}:
            self._render_training_overlay()
        self.statusBar().showMessage(
            f"Showing overview at {self.overview_resolution.value()}% resolution ")
        return True

    def show_full_resolution_roi(self) -> bool:
        roi = self._selected_full_resolution_roi()
        if roi is None or self.current_path is None or not self.current_path.exists():
            QMessageBox.information(
                self,
                "Full-resolution view",
                "Select exactly one Include or Analysis box first.",
            )
            return False
        try:
            image, bounds = load_region(
                self.current_path,
                self._roi_bounds(roi),
                color=True,
                scale_percent=self.roi_resolution.value(),
            )
        except Exception as error:
            QMessageBox.critical(self, "Full-resolution view failed", str(error))
            return False
        self.display_region_bgr = image
        self.display_region_bounds = bounds
        self._set_canvas_image(image, bounds)
        if self.canvas.current_tool in {"seed", "seed_eraser"}:
            self._render_training_overlay()
        self.statusBar().showMessage(
            f"Showing {image.shape[1]} × {image.shape[0]} ROI pixels "
            f"at {self.roi_resolution.value()}% resolution",
            5000,
        )
        return True

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open micrograph",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if path:
            self._load_new_image(Path(path))

    def _load_new_image(self, path: Path) -> None:
        try:
            preview, full_size = load_scaled_image(
                path, self.overview_resolution.value(), color=True
            )
            digest = sha256_file(path)
        except Exception as error:
            QMessageBox.critical(self, "Open failed", str(error))
            return
        self.current_path = path.resolve()
        self.current_project = None
        self.preview_bgr = preview
        self.display_region_bgr = None
        self.display_region_bounds = None
        self.gray = None
        self.analysis_mask = None
        self.current_mask = None
        self.current_labels = None
        self.current_layer_id = None
        self.layer_masks.clear()
        self.particles = []
        self.grouping_preview = None
        self.grouping_preview_definition = None
        self._pending_grouping_preview = None
        if hasattr(self, "grouping_preview_timer"):
            self.grouping_preview_timer.stop()
        self.manifest = ProjectManifest(
            name=path.stem,
            source_path=str(self.current_path),
            source_sha256=digest,
            image_width=full_size[0],
            image_height=full_size[1],
            recipes=[SegmentationRecipe()],
            region_recipes=[RegionClassifierRecipe()],
        )
        self.training_labels = np.zeros(preview.shape[:2], dtype=np.uint16)
        self.run_scope_ids = []
        self._set_canvas_image(preview)
        self._refresh_all()
        self._set_dirty(True)

    def open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open .gstproj directory")
        if not path:
            return
        try:
            manifest, masks = load_project(path)
            source = resolve_source(path, manifest)
            issues = validate_project(path)
            if source.exists():
                preview, _ = load_scaled_image(
                    source, self.overview_resolution.value(), color=True
                )
            else:
                preview = cv2.imread(str(Path(path) / "previews" / "source.jpg"))
                if preview is None:
                    raise FileNotFoundError(
                        "Source image and saved project thumbnail are both missing"
                    )
        except Exception as error:
            QMessageBox.critical(self, "Open failed", str(error))
            return
        if issues:
            QMessageBox.warning(self, "Project validation", "\n".join(issues))
        self.current_project = Path(path)
        self.current_path = source
        self.manifest = manifest
        self.layer_masks = masks
        self.preview_bgr = preview
        self.display_region_bgr = None
        self.display_region_bounds = None
        self.gray = None
        self.analysis_mask = None
        self.current_mask = None
        self.current_labels = None
        self.current_layer_id = None
        self.particles = []
        self.grouping_preview = None
        self.grouping_preview_definition = None
        self._pending_grouping_preview = None
        if hasattr(self, "grouping_preview_timer"):
            self.grouping_preview_timer.stop()
        self.selected_labels.clear()
        self._set_canvas_image(preview)
        particle_layer = next((layer for layer in manifest.layers if layer.kind == "instances"), None)
        if particle_layer and particle_layer.id in masks:
            self.current_layer_id = particle_layer.id
            self.current_labels = masks[particle_layer.id].astype(np.int32)
            self.current_mask = (self.current_labels > 0).astype(np.uint8)
            self.particles = manifest.particle_records.get(particle_layer.id, [])
            domain_layer = next(
                (
                    layer
                    for layer in manifest.layers
                    if layer.kind == "domain"
                    and layer.scope_roi_ids == particle_layer.scope_roi_ids
                ),
                None,
            )
            if domain_layer and domain_layer.id in masks:
                self.analysis_mask = masks[domain_layer.id].astype(bool)
        self._render_visible_layers()
        self.training_labels = np.zeros(preview.shape[:2], dtype=np.uint16)
        self.run_scope_ids = []
        self._restore_training_strokes()
        if self.canvas.current_tool in {"seed", "seed_eraser"}:
            self._render_training_overlay()
        self._refresh_all()
        self._set_dirty(False)

    def save_project_dialog(self) -> None:
        if self.manifest is None:
            return
        target = self.current_project
        if target is None:
            selected, _ = QFileDialog.getSaveFileName(
                self, "Save project", f"{self.manifest.name}.gstproj", "GST projects (*.gstproj)"
            )
            if not selected:
                return
            target = Path(selected)
        try:
            self.current_project = save_project(target, self.manifest, self.layer_masks)
        except Exception as error:
            QMessageBox.critical(self, "Save failed", str(error))
            return
        self._set_dirty(False)
        self.statusBar().showMessage(f"Saved {self.current_project}", 5000)

    def save_portable_project(self) -> None:
        if self.manifest is None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save portable project",
            f"{self.manifest.name}-portable.gstproj",
            "GST projects (*.gstproj)",
        )
        if not selected:
            return
        try:
            self.current_project = save_project(
                selected,
                self.manifest,
                self.layer_masks,
                portable=True,
                source_override=self.current_path,
            )
        except Exception as error:
            QMessageBox.critical(self, "Portable save failed", str(error))
            return
        self._set_dirty(False)

    def relink_source_dialog(self) -> None:
        if self.manifest is None:
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Relink source image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not selected:
            return
        answer = QMessageBox.question(
            self,
            "Create source revision",
            "Relinking invalidates all derived masks and analysis runs. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.manifest = relink_source(self.manifest, selected)
            self.layer_masks.clear()
            self._load_relinked_source(Path(selected))
        except Exception as error:
            QMessageBox.critical(self, "Relink failed", str(error))

    def _load_relinked_source(self, path: Path) -> None:
        preview, _ = load_scaled_image(
            path, self.overview_resolution.value(), color=True
        )
        self.current_path = path.resolve()
        self.preview_bgr = preview
        self.display_region_bgr = None
        self.display_region_bounds = None
        self.gray = None
        self.analysis_mask = None
        self.current_mask = None
        self.current_labels = None
        self.current_layer_id = None
        self.particles = []
        self.training_labels = np.zeros(preview.shape[:2], dtype=np.uint16)
        self._set_canvas_image(preview)
        self._refresh_all()
        self._set_dirty(True)

    def export_dialog(self) -> None:
        if self.manifest is None:
            return
        output = QFileDialog.getExistingDirectory(self, "Choose export directory")
        if not output:
            return
        try:
            fractions = compute_project_fractions(self.manifest, self.layer_masks)
            source = load_image(self.current_path, color=True)
            export_analysis(
                output,
                self.manifest,
                self.layer_masks,
                particles=self.particles,
                fractions=fractions,
                source_image=source,
            )
        except Exception as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return
        self.statusBar().showMessage(f"Exported to {output}", 5000)

    def _recipe_from_controls(self) -> SegmentationRecipe:
        sauvola_window = self.window_size.value()
        if sauvola_window % 2 == 0:
            sauvola_window += 1
            self.window_size.setValue(sauvola_window)
        gaussian_block = self.gaussian_block.value()
        if gaussian_block % 2 == 0:
            gaussian_block += 1
            self.gaussian_block.setValue(gaussian_block)
        return SegmentationRecipe(
            name="Interactive particle segmentation",
            target_class_id=self.binary_class.currentData(),
            channel=self.channel.currentData(),
            segmentation_method=self.segmentation_method.currentData(),
            threshold_method=self.method.currentData(),
            polarity=self.polarity.currentData(),
            illumination_correction=self.illumination.isChecked(),
            rolling_ball_radius_px=self.ball_radius.value(),
            sauvola_window_px=sauvola_window,
            gaussian_block_px=gaussian_block,
            sauvola_k=self.sauvola_k.value(),
            gaussian_c=self.gaussian_c.value(),
            manual_threshold_low=self.manual_threshold_low.value(),
            manual_threshold_high=self.manual_threshold_high.value(),
            gaussian_blur_sigma=self.gaussian_blur_sigma.value(),
            fill_holes=self.fill_holes.isChecked(),
            open_radius_px=self.open_radius.value(),
            close_radius_px=self.close_radius.value(),
            min_particle_area_px=self.min_area.value(),
            max_particle_area_px=self.max_area.value() or None,
            tile_size_px=self.tile_size.value(),
            split_touching=self.split_touching.isChecked(),
            watershed_min_distance_px=self.watershed_distance.value(),
            morphological_input=self.morphological_input.currentData(),
            morphological_gradient=self.morphological_gradient.currentData(),
            morphological_gradient_radius_px=self.morphological_gradient_radius.value(),
            morphological_tolerance=self.morphological_tolerance.value(),
            morphological_connectivity=self.morphological_connectivity.currentData(),
            morphological_calculate_dams=self.morphological_calculate_dams.isChecked(),
        )

    def run_segmentation(self) -> None:
        if self.current_path is None or self.manifest is None or self.worker is not None:
            return
        if not self.current_path.exists():
            QMessageBox.critical(
                self,
                "Source missing",
                "Relink the source image before running analysis.",
            )
            return
        if sha256_file(self.current_path) != self.manifest.source_sha256:
            QMessageBox.critical(
                self,
                "Source changed",
                "The source SHA-256 no longer matches this project. Relink it as a new source revision before analysis.",
            )
            return
        control_recipe = self._recipe_from_controls()
        selected_roi = None
        if self.selected_roi_only.isChecked():
            selected_roi = self._selected_full_resolution_roi()
            if selected_roi is None:
                QMessageBox.information(
                    self,
                    "Selected-ROI analysis",
                    "Select exactly one Include or Analysis box before running "
                    "a selected-ROI analysis.",
                )
                return
        self.run_scope_ids = [selected_roi.id] if selected_roi is not None else []
        confirmed_matches = (
            self.confirmed_preview_recipe is not None
            and self.confirmed_preview_scope_ids == self.run_scope_ids
            and self.confirmed_preview_recipe.model_dump(exclude={"id", "name"})
            == control_recipe.model_dump(exclude={"id", "name"})
        )
        recipe = (
            self.confirmed_preview_recipe.model_copy()
            if confirmed_matches
            else control_recipe
        )
        if len(self.run_scope_ids) == 1:
            roi = next((item for item in self.manifest.rois if item.id == self.run_scope_ids[0]), None)
            if roi is not None:
                if roi.recipe_id:
                    recipe = recipe.model_copy(
                        update={"id": roi.recipe_id, "name": f"{roi.name} recipe"}
                    )
                    existing = next(
                        (
                            index
                            for index, item in enumerate(self.manifest.recipes)
                            if item.id == roi.recipe_id
                        ),
                        None,
                    )
                    if existing is None:
                        self.manifest.recipes.append(recipe)
                    else:
                        self.manifest.recipes[existing] = recipe
                else:
                    recipe = recipe.model_copy(update={"name": f"{roi.name} recipe"})
                    roi.recipe_id = recipe.id
                    self.manifest.recipes.append(recipe)
        else:
            self.manifest.recipes.append(recipe)
        self.run_recipe = recipe
        worker_function = _analyze_source
        worker_arguments = (
            self.current_path,
            list(self.manifest.rois),
            self.run_scope_ids,
            recipe,
            self.manifest.calibration,
        )
        if selected_roi is not None:
            worker_function = _analyze_source_region
            worker_arguments = (
                self.current_path,
                self._roi_bounds(selected_roi),
                list(self.manifest.rois),
                self.run_scope_ids,
                recipe,
                self.manifest.calibration,
            )
        self.worker = FunctionWorker(
            worker_function,
            *worker_arguments,
            with_callbacks=True,
        )
        self.worker.signals.progress.connect(self._progress)
        self.worker.signals.result.connect(self._particle_result)
        self.worker.signals.error.connect(self._worker_error)
        self.worker.signals.finished.connect(self._worker_finished)
        self.preview_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        if selected_roi is not None:
            left, top, right, bottom = self._roi_bounds(selected_roi)
            self.preview_status.setText(
                f"Running selected ROI at original resolution: "
                f"{right - left} × {bottom - top} pixels"
            )
        else:
            self.preview_status.setText(
                f"Running original {self.manifest.image_width} × "
                f"{self.manifest.image_height} pixel source"
            )
        self.thread_pool.start(self.worker)

    def run_model_inference_dialog(self) -> None:
        """Choose a local model pack and run it off the UI thread at source resolution."""
        if self.current_path is None or self.manifest is None or self.worker is not None:
            return
        if not self.current_path.exists() or sha256_file(self.current_path) != self.manifest.source_sha256:
            QMessageBox.critical(
                self,
                "Source changed",
                "Relink the unchanged source as a new revision before running model inference.",
            )
            return
        selected = QFileDialog.getExistingDirectory(self, "Choose model-pack directory")
        if not selected:
            return
        try:
            pack = ModelPack.open(selected)
            recipe = pack.default_recipe()
        except (FileNotFoundError, ValueError) as error:
            QMessageBox.critical(self, "Model pack unavailable", str(error))
            return
        self.manifest.model_inference_recipes.append(recipe)
        self.worker = FunctionWorker(
            _analyze_model_source,
            self.current_path,
            selected,
            recipe,
            self.manifest.calibration,
            with_callbacks=True,
        )
        self.worker.signals.progress.connect(self._progress)
        self.worker.signals.result.connect(self._model_inference_result)
        self.worker.signals.error.connect(self._worker_error)
        self.worker.signals.finished.connect(self._worker_finished)
        self.preview_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.run_model_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.preview_status.setText(
            f"Running model pack {pack.manifest.model_id} at original source resolution"
        )
        self.thread_pool.start(self.worker)

    def _cellpose_recipe_dialog(self, targets: list[ROI]) -> CellposeInferenceRecipe | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Run local Cellpose-SAM v2")
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "Cellpose-SAM v2 runs locally and never uploads a source image. Missing stock "
            "weights are downloaded once. The stock weights are CC-BY-NC: running this "
            "records a non-commercial acknowledgement in the project."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        scope = QLabel(
            f"Scope — {len(targets)} Analysis box{'es' if len(targets) != 1 else ''} at native "
            "resolution. The rest of the overview is never sent to the model."
        )
        scope.setWordWrap(True)
        layout.addWidget(scope)
        layout.addWidget(self._cellpose_scope_list(targets))
        form = QFormLayout()
        modality = QComboBox()
        modality.addItem("Biological cells", "biological")
        modality.addItem("Metallography particles (experimental)", "metallography")
        form.addRow("Result type", modality)
        device = QComboBox()
        device.addItem("NVIDIA CUDA GPU", "gpu")
        device.addItem("CPU", "cpu")
        cuda_ready = _cuda_is_available()
        device.setCurrentIndex(0 if cuda_ready else 1)
        device.setToolTip(
            "Requires a CUDA-enabled PyTorch installation and an NVIDIA GPU visible to PyTorch."
            if cuda_ready
            else "PyTorch cannot see a CUDA device on this machine, so CPU is preselected."
        )
        form.addRow("Execution device", device)
        diameter = QDoubleSpinBox()
        diameter.setRange(0, 100_000)
        diameter.setDecimals(1)
        diameter.setSpecialValueText("Cellpose default (30 px)")
        # Native-resolution Analysis boxes on stitched metallographs run ~46 px particles.
        diameter.setValue(46)
        diameter.setToolTip(
            "Typical particle diameter in source pixels. Cellpose-SAM rescales the region by "
            "30 / diameter, so the default treats particles as ~30 px and under-segments larger "
            "ones. If a downscaled export segments better in the Cellpose app, set this to 30 "
            "divided by that export's scale."
        )
        form.addRow("Diameter (px)", diameter)
        cellprob = QDoubleSpinBox()
        cellprob.setRange(-10, 10)
        cellprob.setDecimals(2)
        cellprob.setValue(0.0)
        form.addRow("Cell probability threshold", cellprob)
        flow = QDoubleSpinBox()
        flow.setRange(0.01, 10)
        flow.setDecimals(2)
        flow.setValue(0.4)
        form.addRow("Flow threshold", flow)
        min_size = QSpinBox()
        min_size.setRange(0, 10_000_000)
        min_size.setValue(15)
        form.addRow("Minimum mask size (px)", min_size)
        overlap = QDoubleSpinBox()
        overlap.setRange(0, 0.95)
        overlap.setSingleStep(0.05)
        overlap.setDecimals(2)
        overlap.setValue(0.1)
        form.addRow("Tile overlap", overlap)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return CellposeInferenceRecipe(
            modality=modality.currentData(),
            device=device.currentData(),
            diameter_px=diameter.value() or None,
            cellprob_threshold=cellprob.value(),
            flow_threshold=flow.value(),
            min_size_px=min_size.value(),
            tile_overlap=overlap.value(),
            noncommercial_license_accepted=True,
        )

    def _cellpose_scope_list(self, targets: list[ROI]) -> QListWidget:
        """Show exactly which Analysis boxes this run will segment, largest first."""
        widget = QListWidget()
        widget.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        rows = []
        for roi in targets:
            left, top, right, bottom = self._roi_bounds(roi)
            rows.append(((right - left) * (bottom - top), roi.name, left, top, right, bottom))
        for pixels, name, left, top, right, bottom in sorted(rows, reverse=True):
            item = QListWidgetItem(
                f"{name} — {right - left} × {bottom - top} px at ({left}, {top}), "
                f"{pixels / 1e6:.2f} Mpx"
            )
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            widget.addItem(item)
        # Fit up to five rows at the theme's own row height, then scroll.
        row_height = widget.sizeHintForRow(0) if rows else 0
        widget.setFixedHeight(
            row_height * min(len(rows), 5) + 2 * widget.frameWidth() + 2
        )
        return widget

    def run_cellpose_inference_dialog(self) -> None:
        """Run stock Cellpose-SAM v2 over the project's Analysis boxes in a cancellable worker."""
        if self.current_path is None or self.manifest is None or self.worker is not None:
            return
        if not self.current_path.exists() or sha256_file(self.current_path) != self.manifest.source_sha256:
            QMessageBox.critical(
                self,
                "Source changed",
                "Relink the unchanged source as a new revision before running Cellpose inference.",
            )
            return
        available = _cellpose_analysis_boxes(self.manifest.rois, ())
        if not available:
            QMessageBox.information(
                self,
                "Analysis box required",
                "Cellpose-SAM runs inside Analysis boxes only, never over the whole overview. "
                "Draw at least one Analysis box before running it.",
            )
            return
        scope_ids: list[str] = []
        targets = available
        if self.selected_roi_only.isChecked():
            selected = self._selected_analysis_box()
            if selected is None:
                QMessageBox.information(
                    self,
                    "Selected-ROI analysis",
                    "Select exactly one Analysis box before running Cellpose.",
                )
                return
            scope_ids = [selected.id]
            targets = [selected]
        recipe = self._cellpose_recipe_dialog(targets)
        if recipe is None:
            return
        self.run_scope_ids = scope_ids
        self.worker = FunctionWorker(
            _analyze_cellpose_source,
            self.current_path,
            list(self.manifest.rois),
            scope_ids,
            recipe,
            self.manifest.calibration,
            True,  # stock weights download once on demand; no per-run confirmation
            with_callbacks=True,
        )
        self.worker.signals.progress.connect(self._progress)
        self.worker.signals.result.connect(self._cellpose_inference_result)
        self.worker.signals.error.connect(self._worker_error)
        self.worker.signals.finished.connect(self._worker_finished)
        self.preview_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.run_model_button.setEnabled(False)
        self.run_cellpose_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        boxes = len(targets)
        self.preview_status.setText(
            f"Running local Cellpose-SAM v2 on {boxes} Analysis box{'es' if boxes != 1 else ''} "
            "at original source resolution"
        )
        self.thread_pool.start(self.worker)

    def _cellpose_inference_result(self, payload) -> None:
        gray, domain, result, recipe = payload
        assert self.manifest is not None
        output_class = "Cell" if recipe.modality == "biological" else "Particle"
        instance_layer = SegmentationLayer(
            name=f"Cellpose-SAM v2 {output_class.lower()}s",
            class_id=self._model_class(output_class, 0).id,
            kind="instances",
            scope_roi_ids=list(self.run_scope_ids),
            review_status="pending",
        )
        domain_layer = SegmentationLayer(
            name="Cellpose analysis domain",
            kind="domain",
            scope_roi_ids=list(self.run_scope_ids),
            visible=False,
            review_status="pending",
        )
        run = CellposeInferenceRun(
            recipe=recipe,
            layer_ids=[instance_layer.id, domain_layer.id],
            summary=result.summary,
            source_bounds_px=result.source_bounds_px,
            region_bounds_px=result.region_bounds_px,
            cellpose_version=result.cellpose_version,
            torch_version=result.torch_version,
            model_sha256=result.model_sha256,
            model_cache_path=result.model_cache_path,
            license_acknowledged_at=datetime.now(UTC),
        )
        instance_layer.source_run_id = run.id
        domain_layer.source_run_id = run.id
        self.manifest.cellpose_inference_recipes.append(recipe)
        self.manifest.cellpose_inference_runs.append(run)
        self.manifest.layers.extend([instance_layer, domain_layer])
        self.manifest.particle_records[instance_layer.id] = result.particles
        self.layer_masks[instance_layer.id] = result.labels
        self.layer_masks[domain_layer.id] = domain.astype(np.uint8)
        self.gray = gray
        self.analysis_mask = domain
        self.current_layer_id = instance_layer.id
        self.current_labels = result.labels
        self.current_mask = (result.labels > 0).astype(np.uint8)
        self.particles = result.particles
        self._render_visible_layers()
        summary = {"review_status": "pending expert review", **result.summary}
        if recipe.modality == "metallography":
            summary["validation_note"] = "Experimental metallography result; validation required"
        self._show_summary(summary)
        self._refresh_all()
        self._set_dirty(True)
        self.preview_status.setText("Cellpose result saved as pending review; inspect overlays before confirmation")

    def _model_class(self, name: str, index: int) -> ClassDefinition:
        assert self.manifest is not None
        existing = next(
            (item for item in self.manifest.classes if item.name.casefold() == name.casefold()),
            None,
        )
        if existing is not None:
            return existing
        colors = ("#66bb6a", "#42a5f5", "#ab47bc", "#ef5350", "#ffb300")
        item = ClassDefinition(name=name, color=colors[index % len(colors)])
        self.manifest.classes.append(item)
        return item

    def _model_inference_result(self, payload) -> None:
        gray, domain, result, pack, recipe = payload
        assert self.manifest is not None
        layers: list[SegmentationLayer] = []
        if result.semantic_labels is not None:
            class_map = {
                value: self._model_class(name, value).id
                for value, name in pack.manifest.semantic_classes.items()
                if value != 0 and name.casefold() != "background"
            }
            semantic_layer = SegmentationLayer(
                name=f"{pack.manifest.model_id} phases",
                kind="multiclass",
                class_value_map=class_map,
                review_status="pending",
            )
            layers.append(semantic_layer)
            self.layer_masks[semantic_layer.id] = result.semantic_labels
        particle_layer = None
        if result.particle_labels is not None:
            particle_layer = SegmentationLayer(
                name=f"{pack.manifest.model_id} particles",
                class_id=self._model_class("Particle", 0).id,
                kind="instances",
                review_status="pending",
            )
            layers.append(particle_layer)
            self.layer_masks[particle_layer.id] = result.particle_labels
            self.manifest.particle_records[particle_layer.id] = result.particles
        domain_layer = SegmentationLayer(
            name="Model analysis domain",
            kind="domain",
            visible=False,
            review_status="pending",
        )
        layers.append(domain_layer)
        self.layer_masks[domain_layer.id] = domain.astype(np.uint8)
        run = ModelInferenceRun(
            recipe=recipe,
            layer_ids=[item.id for item in layers],
            summary=result.summary,
            source_bounds_px=(0, 0, self.manifest.image_width, self.manifest.image_height),
            runtime_version=str(result.summary["runtime_version"]),
        )
        for layer in layers:
            layer.source_run_id = run.id
        self.manifest.layers.extend(layers)
        self.manifest.model_inference_runs.append(run)
        self.gray = gray
        self.analysis_mask = domain
        active_layer = particle_layer or next(
            (item for item in layers if item.kind == "multiclass"), None
        )
        if active_layer is not None:
            self.current_layer_id = active_layer.id
            self.current_labels = self.layer_masks[active_layer.id]
            self.current_mask = (self.current_labels > 0).astype(np.uint8)
            self.particles = result.particles if active_layer is particle_layer else []
        self._render_visible_layers()
        self._show_summary({"review_status": "pending expert review", **result.summary})
        self._refresh_all()
        self._set_dirty(True)
        self.preview_status.setText("Model result saved as pending review; inspect overlays before confirmation")

    def run_preview_segmentation(self) -> None:
        if self.preview_bgr is None or self.manifest is None or self.worker is not None:
            return
        selected_roi_preview = self.preview_resolution.currentData() == "roi_full"
        resolution = (
            self.roi_resolution.value()
            if selected_roi_preview
            else self.overview_resolution.value()
        )
        if resolution == 0:
            QMessageBox.information(
                self,
                "Preview resolution",
                "Choose a resolution above 0% before running segmentation preview.",
            )
            return
        if selected_roi_preview:
            roi = self._selected_full_resolution_roi()
            if roi is None:
                QMessageBox.information(
                    self,
                    "Selected-ROI preview",
                    "Select exactly one Include or Analysis box first.",
                )
                return
            self.selected_roi_only.setChecked(True)
            scope_ids = [roi.id]
            if not self.show_full_resolution_roi():
                return
        else:
            selected = self.rois_list.selectedItems() if self.selected_roi_only.isChecked() else []
            scope_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
            if not self.show_overview():
                return
        recipe = self._recipe_from_controls()
        self._pending_preview_recipe = recipe.model_copy()
        self._pending_preview_scope_ids = list(scope_ids)
        self._pending_preview_full_resolution = selected_roi_preview
        self._pending_preview_resolution = resolution
        if selected_roi_preview:
            self.worker = FunctionWorker(
                _analyze_region_preview,
                self.display_region_bgr.copy(),
                self.display_region_bounds,
                list(self.manifest.rois),
                scope_ids,
                recipe,
                self.manifest.calibration,
                with_callbacks=True,
            )
        else:
            self.worker = FunctionWorker(
                _analyze_preview,
                self.preview_bgr.copy(),
                (self.manifest.image_width, self.manifest.image_height),
                list(self.manifest.rois),
                scope_ids,
                recipe,
                self.manifest.calibration,
                with_callbacks=True,
            )
        self.worker.signals.progress.connect(self._progress)
        self.worker.signals.result.connect(self._preview_result)
        self.worker.signals.error.connect(self._worker_error)
        self.worker.signals.finished.connect(self._worker_finished)
        self.preview_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.full_resolution_button.setEnabled(False)
        self.overview_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.thread_pool.start(self.worker)

    def _preview_result(self, result) -> None:
        self.canvas.set_mask_overlay(result.mask)
        pending_recipe = self._pending_preview_recipe
        pending_scope_ids = list(self._pending_preview_scope_ids)
        was_full_resolution = self._pending_preview_full_resolution
        pending_resolution = self._pending_preview_resolution
        current_recipe = self._recipe_from_controls()
        selected = self.rois_list.selectedItems() if self.selected_roi_only.isChecked() else []
        current_scope_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
        unchanged = (
            pending_recipe is not None
            and pending_scope_ids == current_scope_ids
            and pending_resolution
            == (
                self.roi_resolution.value()
                if was_full_resolution
                else self.overview_resolution.value()
            )
            and pending_recipe.model_dump(exclude={"id", "name"})
            == current_recipe.model_dump(exclude={"id", "name"})
        )
        if unchanged:
            self.confirmed_preview_recipe = pending_recipe
            self.confirmed_preview_scope_ids = pending_scope_ids
        else:
            self.confirmed_preview_recipe = None
            self.confirmed_preview_scope_ids = []
        self._pending_preview_recipe = None
        self._pending_preview_scope_ids = []
        self._pending_preview_full_resolution = False
        self._pending_preview_resolution = None
        mode = (
            f"Selected ROI preview ({self.roi_resolution.value()}%)"
            if was_full_resolution
            else f"Overview preview ({self.overview_resolution.value()}%)"
        )
        self._show_summary({"mode": f"{mode} — not saved", **result.summary})
        if unchanged:
            self.preview_status.setText(
                "Preview parameters confirmed — final run will use the original image resolution"
            )
            self.statusBar().showMessage(
                "Preview confirmed; run particles for saved full-resolution results", 6000
            )
        else:
            self.preview_status.setText("Parameters changed during preview — preview again to confirm")
            self.statusBar().showMessage("Preview complete, but its parameters changed", 6000)

    def cancel_analysis(self) -> None:
        if self.worker:
            self.worker.cancel()

    def _progress(self, value: float, message: str) -> None:
        self.progress.setValue(round(value * 1000))
        self.statusBar().showMessage(message)

    def _particle_result(self, payload) -> None:
        gray, domain, result, bounds = payload
        left, top, right, bottom = bounds
        full_width = self.manifest.image_width
        full_height = self.manifest.image_height
        cropped_run = bounds != (0, 0, full_width, full_height)
        if cropped_run:
            expected_shape = (bottom - top, right - left)
            if domain.shape != expected_shape or result.labels.shape != expected_shape:
                raise ValueError(
                    "Selected-ROI result dimensions do not match its source bounds"
                )
            full_domain = np.zeros((full_height, full_width), dtype=bool)
            full_domain[top:bottom, left:right] = domain
            full_labels = np.zeros((full_height, full_width), dtype=np.int32)
            full_labels[top:bottom, left:right] = result.labels
            result.labels = full_labels
            result.mask = full_labels > 0
            scale = (
                self.manifest.calibration.mm_per_pixel
                if self.manifest.calibration
                else None
            )
            result.particles = [
                particle.model_copy(
                    update={
                        "centroid_x_px": particle.centroid_x_px + left,
                        "centroid_y_px": particle.centroid_y_px + top,
                        "centroid_x_mm": (
                            particle.centroid_x_mm + left * scale
                            if particle.centroid_x_mm is not None and scale is not None
                            else None
                        ),
                        "centroid_y_mm": (
                            particle.centroid_y_mm + top * scale
                            if particle.centroid_y_mm is not None and scale is not None
                            else None
                        ),
                    }
                )
                for particle in result.particles
            ]
            domain = full_domain
        result.summary.update(
            {
                "source_width_px": full_width,
                "source_height_px": full_height,
                "processed_width_px": gray.shape[1],
                "processed_height_px": gray.shape[0],
                "analysis_scope": "selected_roi" if cropped_run else "full_image",
                "analysis_resolution": "original",
            }
        )
        self.gray = gray
        self.analysis_mask = domain
        self.current_mask = result.mask.astype(np.uint8)
        self.current_labels = result.labels
        self.particles = result.particles
        target_class = next(
            (
                item
                for item in self.manifest.classes
                if item.id == self.run_recipe.target_class_id
            ),
            next(item for item in self.manifest.classes if item.preset == "particle"),
        )
        layer = next(
            (
                item
                for item in self.manifest.layers
                if item.kind == "instances"
                and item.class_id == target_class.id
                and item.scope_roi_ids == self.run_scope_ids
            ),
            None,
        )
        if layer is None:
            scope_roi = next(
                (
                    roi
                    for roi in self.manifest.rois
                    if self.run_scope_ids == [roi.id]
                ),
                None,
            )
            suffix = f" — {scope_roi.name}" if scope_roi is not None else ""
            layer = SegmentationLayer(
                name=f"{target_class.name}{suffix}",
                class_id=target_class.id,
                kind="instances",
                scope_roi_ids=list(self.run_scope_ids),
            )
            self.manifest.layers.append(layer)
        domain_layer = next(
            (
                item
                for item in self.manifest.layers
                if item.kind == "domain" and item.scope_roi_ids == self.run_scope_ids
            ),
            None,
        )
        if domain_layer is None:
            domain_layer = SegmentationLayer(
                name=f"{layer.name} analysis domain",
                kind="domain",
                scope_roi_ids=list(self.run_scope_ids),
                visible=False,
            )
            self.manifest.layers.append(domain_layer)
        run = AnalysisRun(
            recipe=self.run_recipe,
            layer_ids=[layer.id, domain_layer.id],
            summary=result.summary,
        )
        layer.source_run_id = run.id
        domain_layer.source_run_id = run.id
        self.manifest.runs.append(run)
        self.layer_masks[layer.id] = self.current_labels
        self.layer_masks[domain_layer.id] = self.analysis_mask.astype(np.uint8)
        self.current_layer_id = layer.id
        active_grouping = self._active_particle_grouping(layer.id)
        if active_grouping:
            self.grouping_preview_definition = active_grouping.model_copy(deep=True)
            self.grouping_preview = evaluate_particle_grouping(
                self.particles, active_grouping
            )
            self.particles = self.grouping_preview.particles
        else:
            self.grouping_preview = None
            self.grouping_preview_definition = None
        self.manifest.particle_records[layer.id] = self.particles
        self._render_visible_layers()
        self._show_summary(result.summary)
        self._refresh_all()
        self._set_dirty(True)
        scope = "Selected-ROI" if cropped_run else "Full-image"
        self.preview_status.setText(
            f"{scope} original-resolution analysis complete "
            f"({gray.shape[1]} × {gray.shape[0]} processed px)"
        )

    def _worker_error(self, detail: str) -> None:
        self._pending_preview_recipe = None
        self._pending_preview_scope_ids = []
        self._pending_preview_full_resolution = False
        self._pending_preview_resolution = None
        if detail != "Analysis cancelled":
            QMessageBox.critical(self, "Analysis failed", detail)
        self.statusBar().showMessage(detail, 5000)

    def _worker_finished(self) -> None:
        self.worker = None
        self.preview_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.run_model_button.setEnabled(True)
        self.run_cellpose_button.setEnabled(True)
        self.full_resolution_button.setEnabled(True)
        self.overview_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _line_finished(self, tool: str, start: QPointF, end: QPointF) -> None:
        if self.manifest is None:
            return
        first, last = (start.x(), start.y()), (end.x(), end.y())
        if tool == "calibrate":
            length, ok = QInputDialog.getDouble(
                self, "Scale calibration", "Known scale-bar length", 1.0, 0.000001, 1_000_000, 6
            )
            if not ok:
                return
            unit, ok = QInputDialog.getItem(self, "Scale calibration", "Unit", ["mm", "µm"], 0, False)
            if ok:
                self.manifest.calibration = Calibration.from_reference(first, last, length, unit)
                for index, measurement in enumerate(self.manifest.measurements):
                    self.manifest.measurements[index] = measurement.model_copy(
                        update={
                            "length_mm": measurement.length_px
                            * self.manifest.calibration.mm_per_pixel
                        }
                    )
                self._set_dirty(True)
                self._refresh_measurements()
        elif tool == "measure":
            name, ok = QInputDialog.getText(
                self, "Measurement", "Measurement name", text=f"Measurement {len(self.manifest.measurements) + 1}"
            )
            if ok:
                self.manifest.measurements.append(
                    create_measurement(name, first, last, self.manifest.calibration)
                )
                self._set_dirty(True)
                self._refresh_measurements()
        elif tool == "split" and self.current_labels is not None:
            if len(self.selected_labels) != 1:
                QMessageBox.information(self, "Split particle", "Select exactly one particle first.")
                return
            self.current_labels = split_particle_by_line(
                self.current_labels, next(iter(self.selected_labels)), first, last, max(1, self.brush_radius.value() // 4)
            )
            self._labels_edited("split_particle")

    def _rectangle_finished(self, tool: str, start: QPointF, end: QPointF) -> None:
        kind = ROIKind(tool)
        local_recipe = self._recipe_from_controls() if kind == ROIKind.ANALYSIS_BOX else None
        if local_recipe is not None:
            local_recipe.name = f"Analysis box {len(self.manifest.rois) + 1} recipe"
            self.manifest.recipes.append(local_recipe)
        roi = ROI(
            name=f"{kind.value.replace('_', ' ').title()} {len(self.manifest.rois) + 1}",
            kind=kind,
            shape="rectangle",
            points=[Point(x=start.x(), y=start.y()), Point(x=end.x(), y=end.y())],
            recipe_id=local_recipe.id if local_recipe else None,
        )
        self.manifest.rois.append(roi)
        self.manifest.edits.append(EditEvent(action="add_roi", target_id=roi.id))
        self._refresh_rois()
        self._set_dirty(True)

    def _polygon_finished(self, _tool: str, points: list[QPointF]) -> None:
        roi = ROI(
            name=f"Polygon {len(self.manifest.rois) + 1}",
            kind=ROIKind.INCLUDE,
            shape="polygon",
            points=[Point(x=point.x(), y=point.y()) for point in points],
        )
        self.manifest.rois.append(roi)
        self.manifest.edits.append(EditEvent(action="add_roi", target_id=roi.id))
        self._refresh_rois()
        self._set_dirty(True)

    def _brush_stroke(self, tool: str, points: list[QPointF]) -> None:
        if self.manifest is None:
            return
        if tool in {"seed", "seed_eraser"}:
            self._paint_training(points, erase=tool == "seed_eraser")
        elif tool in {"mask_brush", "mask_eraser"}:
            layer = self._current_editable_layer()
            if layer and layer.kind == "multiclass" and self.current_labels is not None:
                class_id = self.seed_class.currentData()
                value = next(
                    (
                        label
                        for label, mapped_class in layer.class_value_map.items()
                        if mapped_class == class_id
                    ),
                    None,
                )
                if value is None and tool == "mask_brush" and class_id is not None:
                    value = max(layer.class_value_map, default=0) + 1
                    layer.class_value_map[value] = class_id
                if value is not None or tool == "mask_eraser":
                    self.undo_stack.push(
                        RegionPaintCommand(self, points, value or 0, tool == "mask_eraser")
                    )
            elif layer is not None and self.current_mask is not None:
                self.undo_stack.push(MaskLineCommand(self, points, tool == "mask_eraser"))
            else:
                self.statusBar().showMessage(
                    "Nothing to paint. Run particles or region classification, "
                    "then select the result layer.",
                    6000,
                )

    def _paint_training(self, points: list[QPointF], *, erase: bool = False) -> None:
        if self.training_labels is None or self.preview_bgr is None:
            return
        trainable = self._trainable_classes()
        class_index = self.seed_class.currentIndex()
        if class_index < 0:
            return
        label = 0 if erase else class_index + 1
        sx = self.preview_bgr.shape[1] / self.manifest.image_width
        sy = self.preview_bgr.shape[0] / self.manifest.image_height
        coords = [(round(point.x() * sx), round(point.y() * sy)) for point in points]
        radius = max(1, round(self.brush_radius.value() * (sx + sy) / 2))
        if len(coords) == 1:
            cv2.circle(self.training_labels, coords[0], radius, label, cv2.FILLED)
        else:
            cv2.line(self.training_labels, coords[0], coords[-1], label, 2 * radius + 1)
        class_id = trainable[class_index].id
        self.manifest.training_strokes.append(
            TrainingStroke(
                class_id=class_id,
                points=[Point(x=point.x(), y=point.y()) for point in points],
                radius_px=self.brush_radius.value(),
                erase=erase,
            )
        )
        colors = {index + 1: self._class_rgb(item.color) for index, item in enumerate(trainable)}
        self.canvas.set_label_overlay(
            self.training_labels, colors, 0.35, align_to_source=True
        )
        self._set_dirty(True)

    def _render_training_overlay(self) -> None:
        """Render overview-sized class seeds at their full-source positions."""
        if self.training_labels is None:
            return
        colors = {
            index + 1: self._class_rgb(item.color)
            for index, item in enumerate(self._trainable_classes())
        }
        self.canvas.set_label_overlay(
            self.training_labels, colors, 0.35, align_to_source=True
        )

    def _restore_training_strokes(self) -> None:
        if self.training_labels is None or self.preview_bgr is None:
            return
        trainable = self._trainable_classes()
        class_indexes = {item.id: index + 1 for index, item in enumerate(trainable)}
        sx = self.preview_bgr.shape[1] / self.manifest.image_width
        sy = self.preview_bgr.shape[0] / self.manifest.image_height
        for stroke in self.manifest.training_strokes:
            label = 0 if stroke.erase else class_indexes.get(stroke.class_id)
            if label is None or not stroke.points:
                continue
            coords = [(round(p.x * sx), round(p.y * sy)) for p in stroke.points]
            radius = max(1, round(stroke.radius_px * (sx + sy) / 2))
            if len(coords) == 1:
                cv2.circle(self.training_labels, coords[0], radius, label, cv2.FILLED)
            else:
                cv2.line(self.training_labels, coords[0], coords[-1], label, 2 * radius + 1)

    def train_regions(self) -> None:
        if self.manifest is None or self.worker is not None:
            return
        roi = self._selected_analysis_box()
        if roi is None:
            QMessageBox.information(
                self,
                "Region-classification Analysis box",
                "Select exactly one Analysis box before training the region classifier.",
            )
            return
        if self.current_path is None or not self.current_path.exists():
            QMessageBox.critical(
                self,
                "Source missing",
                "Relink the source image before training region classification.",
            )
            return
        resolution = self.roi_resolution.value()
        if resolution == 0:
            QMessageBox.information(
                self,
                "Selected ROI resolution",
                "Choose a Selected ROI resolution above 0% before training.",
            )
            return
        bounds = self._roi_bounds(roi)
        left, top, right, bottom = bounds
        width = max(1, round((right - left) * resolution / 100))
        height = max(1, round((bottom - top) * resolution / 100))
        pixel_count = width * height
        if pixel_count > REGION_CLASSIFICATION_MAX_PIXELS:
            recommended = self._maximum_region_roi_resolution(bounds)
            QMessageBox.warning(
                self,
                "Region-classification ROI is too large",
                f"The selected Analysis box would be {width} x {height} pixels "
                f"({pixel_count / 1_000_000:.2f} MP). Region classification is "
                "limited to 4.5 MP to control memory use.\n\n"
                f"Set Selected ROI resolution to {recommended}% or lower and train "
                "again. Existing class-seed strokes will be rescaled automatically.",
            )
            return
        trainable = self._trainable_classes()
        training_labels = _region_training_labels(
            self.manifest.training_strokes,
            [item.id for item in trainable],
            bounds,
            (height, width),
        )
        painted_classes = np.unique(training_labels[training_labels > 0])
        if len(painted_classes) < 2:
            QMessageBox.information(
                self,
                "More class seeds required",
                "Paint training strokes for at least two region classes inside the "
                "selected Analysis box.",
            )
            return
        recipe = (
            self.manifest.region_recipes[-1]
            if self.manifest.region_recipes
            else RegionClassifierRecipe()
        )
        worker = FunctionWorker(
            _classify_source_region,
            self.current_path,
            bounds,
            resolution,
            training_labels,
            recipe,
            roi.id,
            with_callbacks=True,
        )
        self.worker = worker
        worker.signals.progress.connect(self._progress)
        worker.signals.result.connect(self._region_result)
        worker.signals.error.connect(self._worker_error)
        worker.signals.finished.connect(self._worker_finished)
        self.preview_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.thread_pool.start(worker)

    @staticmethod
    def _maximum_region_roi_resolution(
        bounds: tuple[int, int, int, int]
    ) -> int:
        left, top, right, bottom = bounds
        for percent in range(100, 0, -1):
            width = max(1, round((right - left) * percent / 100))
            height = max(1, round((bottom - top) * percent / 100))
            if width * height <= REGION_CLASSIFICATION_MAX_PIXELS:
                return percent
        return 1

    def _region_result(self, payload) -> None:
        result, bounds, roi_id = payload
        left, top, right, bottom = bounds
        roi_labels = cv2.resize(
            result.labels.astype(np.uint16),
            (right - left, bottom - top),
            interpolation=cv2.INTER_NEAREST,
        )
        full_labels = np.zeros(
            (self.manifest.image_height, self.manifest.image_width), dtype=np.uint16
        )
        full_labels[top:bottom, left:right] = roi_labels
        layer = next(
            (
                item
                for item in self.manifest.layers
                if item.kind == "multiclass" and item.scope_roi_ids == [roi_id]
            ),
            None,
        )
        trainable = self._trainable_classes()
        mapping = {index + 1: item.id for index, item in enumerate(trainable)}
        roi = next((item for item in self.manifest.rois if item.id == roi_id), None)
        if layer is None:
            layer = SegmentationLayer(
                name=f"Region classification — {roi.name if roi else roi_id[:8]}",
                kind="multiclass",
                scope_roi_ids=[roi_id],
                class_value_map=mapping,
            )
            self.manifest.layers.append(layer)
        else:
            layer.class_value_map = mapping
        result.summary.update(
            {
                "analysis_scope": "selected_roi",
                "scope_roi_id": roi_id,
                "processed_width_px": result.labels.shape[1],
                "processed_height_px": result.labels.shape[0],
                "source_bounds": [left, top, right, bottom],
            }
        )
        self.layer_masks[layer.id] = full_labels
        self.current_layer_id = layer.id
        self.current_labels = full_labels.astype(np.int32)
        self.current_mask = None
        self.particles = []
        self.grouping_preview = None
        self.grouping_preview_definition = None
        self.manifest.edits.append(
            EditEvent(
                action="classify_regions",
                target_id=layer.id,
                details={"scope_roi_id": roi_id},
            )
        )
        self._render_visible_layers()
        self._show_summary(result.summary)
        self._refresh_all()
        self._set_dirty(True)

    def _select_particle(self, point: QPointF) -> None:
        if self.canvas.current_tool == "eyedropper":
            self._pick_image_value(point)
            return
        if self.current_labels is None:
            return
        x = int(np.clip(round(point.x()), 0, self.current_labels.shape[1] - 1))
        y = int(np.clip(round(point.y()), 0, self.current_labels.shape[0] - 1))
        label = int(self.current_labels[y, x])
        if label <= 0:
            self.selected_labels.clear()
        elif QApplication_keyboard_control():
            self.selected_labels.symmetric_difference_update({label})
        else:
            self.selected_labels = {label}
        self.statusBar().showMessage(f"Selected particles: {sorted(self.selected_labels)}", 4000)

    def _pick_image_value(self, point: QPointF) -> None:
        """Apply a robust native-resolution color sample to the selected target."""
        if (
            self.current_path is None
            or not self.current_path.exists()
            or self.manifest is None
        ):
            self.statusBar().showMessage("The source image is unavailable for sampling", 4000)
            return
        x = int(np.clip(round(point.x()), 0, self.manifest.image_width - 1))
        y = int(np.clip(round(point.y()), 0, self.manifest.image_height - 1))
        try:
            sample, _ = load_region(
                self.current_path,
                (x - 2, y - 2, x + 3, y + 3),
                color=True,
            )
        except Exception as error:
            self.statusBar().showMessage(f"Could not sample source image: {error}", 5000)
            return
        bgr = np.rint(np.median(sample.reshape(-1, 3), axis=0)).astype(np.uint8)
        blue, green, red = (int(value) for value in bgr)
        target = self.eyedropper_target.currentData()
        if target in {"threshold_low", "threshold_high"}:
            sampled_channel = to_gray(sample, self.channel.currentData())
            threshold = round(float(np.median(sampled_channel)))
            if target == "threshold_low":
                self.manual_threshold_low.setValue(threshold)
                bound = "lower"
            else:
                self.manual_threshold_high.setValue(threshold)
                bound = "upper"
            self.method.setCurrentIndex(self.method.findData(ThresholdMethod.MANUAL))
            detail = f"manual {bound} threshold {threshold}"
        else:
            class_id = (
                self.binary_class.currentData()
                if target == "binary_color"
                else self.seed_class.currentData()
            )
            class_definition = next(
                (item for item in self.manifest.classes if item.id == class_id), None
            )
            if class_definition is None:
                self.statusBar().showMessage("Select a segmentation class first", 4000)
                return
            class_definition.color = f"#{red:02x}{green:02x}{blue:02x}"
            self._render_visible_layers()
            self._set_dirty(True)
            detail = f"{class_definition.name} color {class_definition.color}"
        self.statusBar().showMessage(
            f"Sampled RGB ({red}, {green}, {blue}) at ({x}, {y}); set {detail}",
            6000,
        )

    def merge_selected(self) -> None:
        if self.current_labels is None or len(self.selected_labels) < 2:
            return
        self.current_labels = merge_particle_labels(self.current_labels, list(self.selected_labels))
        self.selected_labels.clear()
        self._labels_edited("merge_particles")

    def schedule_mask_commit(self) -> None:
        self.mask_commit_timer.start()
        self.canvas.set_mask_overlay(self.current_mask)

    def _commit_mask_edits(self) -> None:
        if self.current_mask is None:
            return
        _, labels = cv2.connectedComponents(self.current_mask.astype(np.uint8), connectivity=8)
        self.current_labels = labels.astype(np.int32)
        self._labels_edited("paint_mask")

    def _labels_edited(self, action: str) -> None:
        self.current_mask = (self.current_labels > 0).astype(np.uint8)
        self.particles = measure_particles(
            self.current_labels, self.manifest.calibration, self.analysis_mask
        )
        layer = next(
            (item for item in self.manifest.layers if item.id == self.current_layer_id), None
        )
        if layer:
            active_grouping = self._active_particle_grouping(layer.id)
            if active_grouping:
                self.grouping_preview_definition = active_grouping.model_copy(deep=True)
                self.grouping_preview = evaluate_particle_grouping(
                    self.particles, active_grouping
                )
                self.particles = self.grouping_preview.particles
            else:
                self.grouping_preview = None
                self.grouping_preview_definition = None
            self.layer_masks[layer.id] = self.current_labels
            self.manifest.particle_records[layer.id] = self.particles
            self.manifest.edits.append(EditEvent(action=action, target_id=layer.id))
        self._render_visible_layers()
        self._refresh_particles()
        self._set_dirty(True)

    def _commit_region_edit(self) -> None:
        layer = next(
            (item for item in self.manifest.layers if item.id == self.current_layer_id), None
        )
        if layer is None or layer.kind != "multiclass":
            return
        self.layer_masks[layer.id] = self.current_labels
        self.manifest.edits.append(EditEvent(action="paint_region", target_id=layer.id))
        self._render_visible_layers()
        self._set_dirty(True)

    def _remove_layers(self, requested_ids: set[str]) -> set[str]:
        """Remove layers and all results from the same analysis runs."""
        if self.manifest is None or not requested_ids:
            return set()
        run_ids = {
            run.id
            for run in self.manifest.runs
            if requested_ids.intersection(run.layer_ids)
        }
        model_run_ids = {
            run.id
            for run in self.manifest.model_inference_runs
            if requested_ids.intersection(run.layer_ids)
        }
        cellpose_run_ids = {
            run.id
            for run in self.manifest.cellpose_inference_runs
            if requested_ids.intersection(run.layer_ids)
        }
        removed_ids = set(requested_ids)
        for run in self.manifest.runs:
            if run.id in run_ids:
                removed_ids.update(run.layer_ids)
        for run in self.manifest.model_inference_runs:
            if run.id in model_run_ids:
                removed_ids.update(run.layer_ids)
        for run in self.manifest.cellpose_inference_runs:
            if run.id in cellpose_run_ids:
                removed_ids.update(run.layer_ids)
        removed_ids.update(
            layer.id
            for layer in self.manifest.layers
            if layer.source_run_id in run_ids | model_run_ids | cellpose_run_ids
        )
        self.manifest.layers = [
            layer for layer in self.manifest.layers if layer.id not in removed_ids
        ]
        self.manifest.runs = [run for run in self.manifest.runs if run.id not in run_ids]
        self.manifest.model_inference_runs = [
            run for run in self.manifest.model_inference_runs if run.id not in model_run_ids
        ]
        self.manifest.cellpose_inference_runs = [
            run for run in self.manifest.cellpose_inference_runs if run.id not in cellpose_run_ids
        ]
        for layer_id in removed_ids:
            self.layer_masks.pop(layer_id, None)
            self.manifest.particle_records.pop(layer_id, None)
            self.manifest.active_particle_groupings.pop(layer_id, None)
        self.manifest.particle_groupings = [
            grouping
            for grouping in self.manifest.particle_groupings
            if grouping.source_layer_id not in removed_ids
        ]
        if self.current_layer_id in removed_ids:
            self.current_layer_id = None
            self.current_mask = None
            self.current_labels = None
            self.analysis_mask = None
            self.particles = []
            self.grouping_preview = None
            self.grouping_preview_definition = None
            self.selected_labels.clear()
        return removed_ids

    def delete_selected_layers(self) -> None:
        if self.manifest is None:
            return
        selected_ids = {
            item.data(Qt.ItemDataRole.UserRole) for item in self.layers_list.selectedItems()
        }
        removed_ids = self._remove_layers(selected_ids)
        for layer_id in removed_ids:
            self.manifest.edits.append(EditEvent(action="delete_layer", target_id=layer_id))
        if removed_ids:
            self._refresh_all()
            self._render_visible_layers()
            self._set_dirty(True)

    def copy_current_roi_mask(self) -> None:
        """Copy the active layer's native-resolution ROI mask to the clipboard."""
        if self.manifest is None or self.current_layer_id is None:
            self.statusBar().showMessage("Select a result layer to copy its mask", 4000)
            return
        layer = next(
            (
                item
                for item in self.manifest.layers
                if item.id == self.current_layer_id
            ),
            None,
        )
        values = self.layer_masks.get(self.current_layer_id)
        if layer is None or values is None:
            self.statusBar().showMessage("The selected layer has no saved mask", 4000)
            return
        roi = self._selected_full_resolution_roi()
        if roi is None and len(layer.scope_roi_ids) == 1:
            roi = next(
                (
                    item
                    for item in self.manifest.rois
                    if item.id == layer.scope_roi_ids[0]
                ),
                None,
            )
        if roi is not None:
            left, top, right, bottom = self._roi_bounds(roi)
            values = values[top:bottom, left:right]
            scope_name = roi.name
        else:
            scope_name = "full image"
        mask = np.ascontiguousarray((values > 0).astype(np.uint8) * 255)
        height, width = mask.shape[:2]
        image = QImage(
            mask.data,
            width,
            height,
            mask.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        QApplication.clipboard().setImage(image)
        self.statusBar().showMessage(
            f"Copied {layer.name} mask for {scope_name} ({width} × {height} px)",
            5000,
        )

    def delete_selected_rois(self) -> None:
        if self.manifest is None:
            return
        selected_ids = {
            item.data(Qt.ItemDataRole.UserRole) for item in self.rois_list.selectedItems()
        }
        if not selected_ids:
            return
        removed_recipe_ids = {
            roi.recipe_id
            for roi in self.manifest.rois
            if roi.id in selected_ids and roi.recipe_id is not None
        }
        dependent_layers = {
            layer.id
            for layer in self.manifest.layers
            if selected_ids.intersection(layer.scope_roi_ids)
        }
        removed_layers = self._remove_layers(dependent_layers)
        self.manifest.rois = [roi for roi in self.manifest.rois if roi.id not in selected_ids]
        still_used_recipes = {roi.recipe_id for roi in self.manifest.rois if roi.recipe_id}
        self.manifest.recipes = [
            recipe
            for recipe in self.manifest.recipes
            if recipe.id not in removed_recipe_ids or recipe.id in still_used_recipes
        ]
        self.run_scope_ids = [value for value in self.run_scope_ids if value not in selected_ids]
        for roi_id in selected_ids:
            self.manifest.edits.append(EditEvent(action="delete_roi", target_id=roi_id))
        for layer_id in removed_layers:
            self.manifest.edits.append(EditEvent(action="delete_layer", target_id=layer_id))
        self.display_region_bgr = None
        self.display_region_bounds = None
        self._invalidate_preview_confirmation()
        self._refresh_all()
        self.show_overview()
        self._set_dirty(True)

    def delete_selected_measurements(self) -> None:
        if self.manifest is None:
            return
        selected_ids = {
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.measurements_list.selectedItems()
        }
        if not selected_ids:
            return
        if "__calibration__" in selected_ids:
            self.manifest.calibration = None
            self.manifest.measurements = [
                measurement.model_copy(update={"length_mm": None})
                for measurement in self.manifest.measurements
            ]
            self.manifest.edits.append(EditEvent(action="delete_calibration"))
        measurement_ids = selected_ids - {"__calibration__"}
        self.manifest.measurements = [
            measurement
            for measurement in self.manifest.measurements
            if measurement.id not in measurement_ids
        ]
        for measurement_id in measurement_ids:
            self.manifest.edits.append(
                EditEvent(action="delete_measurement", target_id=measurement_id)
            )
        self._refresh_measurements()
        self._set_dirty(True)

    def delete_selected_groups(self) -> None:
        """Compatibility wrapper for the former group-table action."""
        self.grouping_panel._remove_group()

    def add_group(self) -> None:
        """Compatibility wrapper for the former group-table action."""
        self.grouping_panel._add_group()

    def apply_groups(self) -> None:
        """Compatibility wrapper: save the current draft as a grouping scheme."""
        grouping = self.grouping_panel.current_grouping()
        if grouping:
            self._save_particle_grouping(grouping)

    def _active_particle_grouping(
        self, layer_id: str | None = None
    ) -> ParticleGrouping | None:
        if self.manifest is None:
            return None
        layer_id = layer_id or self.current_layer_id
        if layer_id is None:
            return None
        grouping_id = self.manifest.active_particle_groupings.get(layer_id)
        return next(
            (
                grouping
                for grouping in self.manifest.particle_groupings
                if grouping.id == grouping_id and grouping.source_layer_id == layer_id
            ),
            None,
        )

    def _queue_grouping_preview(self, grouping: ParticleGrouping) -> None:
        if grouping.source_layer_id != self.current_layer_id:
            return
        self._pending_grouping_preview = grouping
        self.grouping_preview_timer.start()

    def _apply_pending_grouping_preview(self) -> None:
        grouping = self._pending_grouping_preview
        self._pending_grouping_preview = None
        if grouping is not None:
            self._preview_particle_grouping(grouping)

    def _preview_particle_grouping(self, grouping: ParticleGrouping) -> None:
        if (
            self.manifest is None
            or self.current_layer_id is None
            or grouping.source_layer_id != self.current_layer_id
        ):
            return
        source_particles = self.manifest.particle_records.get(
            self.current_layer_id, self.particles
        )
        result = evaluate_particle_grouping(
            source_particles, grouping, copy_records=False
        )
        self.grouping_preview_definition = grouping.model_copy(deep=True)
        self.grouping_preview = result
        analyzed_pixels, exclude_border = self._grouping_statistics_context()
        statistics = particle_group_statistics(
            source_particles,
            grouping,
            result,
            analyzed_pixels=analyzed_pixels,
            exclude_border_from_size=exclude_border,
        )
        self.grouping_panel.set_statistics(statistics)
        self.grouping_panel.set_preview_counts(
            len(result.included_labels),
            len(source_particles),
            len(result.unclassified_labels),
        )
        self._refresh_particles()
        self._render_visible_layers()

    def _grouping_statistics_context(self) -> tuple[int | None, bool]:
        if self.manifest is None or self.current_layer_id is None:
            return None, True
        layer = next(
            (item for item in self.manifest.layers if item.id == self.current_layer_id),
            None,
        )
        run = next(
            (
                item
                for item in self.manifest.runs
                if layer is not None and item.id == layer.source_run_id
            ),
            None,
        )
        if run is not None:
            return (
                run.summary.get("analyzed_pixels"),
                run.recipe.exclude_border_particles_from_size_stats,
            )
        model_run = next(
            (
                item
                for item in self.manifest.model_inference_runs
                if layer is not None and item.id == layer.source_run_id
            ),
            None,
        )
        if model_run is not None:
            return model_run.summary.get("analyzed_pixels"), True
        cellpose_run = next(
            (
                item
                for item in self.manifest.cellpose_inference_runs
                if layer is not None and item.id == layer.source_run_id
            ),
            None,
        )
        return (cellpose_run.summary.get("analyzed_pixels"), True) if cellpose_run else (None, True)

    def _save_particle_grouping(self, grouping: ParticleGrouping) -> None:
        if self.manifest is None or self.current_layer_id is None:
            return
        self.grouping_preview_timer.stop()
        self._pending_grouping_preview = None
        grouping.source_layer_id = self.current_layer_id
        existing = next(
            (
                index
                for index, value in enumerate(self.manifest.particle_groupings)
                if value.id == grouping.id
            ),
            None,
        )
        if existing is None:
            self.manifest.particle_groupings.append(grouping)
        else:
            self.manifest.particle_groupings[existing] = grouping
        self.manifest.active_particle_groupings[self.current_layer_id] = grouping.id
        source_particles = self.manifest.particle_records.get(
            self.current_layer_id, self.particles
        )
        self.grouping_preview_definition = grouping.model_copy(deep=True)
        self.grouping_preview = evaluate_particle_grouping(source_particles, grouping)
        self.particles = self.grouping_preview.particles
        self.manifest.particle_records[self.current_layer_id] = self.particles
        self.manifest.edits.append(
            EditEvent(
                action="save_particle_grouping",
                target_id=grouping.id,
                details={"source_layer_id": self.current_layer_id},
            )
        )
        self._refresh_grouping_panel()
        self._refresh_particles()
        self._render_visible_layers()
        self._set_dirty(True)
        self.statusBar().showMessage(
            f"Saved particle grouping {grouping.name!r} to the project; "
            "save the project to write it to disk",
            5000,
        )

    def _delete_particle_grouping(self, grouping_id: str) -> None:
        if self.manifest is None:
            return
        self.grouping_preview_timer.stop()
        self._pending_grouping_preview = None
        removed = next(
            (
                grouping
                for grouping in self.manifest.particle_groupings
                if grouping.id == grouping_id
            ),
            None,
        )
        if removed is None:
            return
        self.manifest.particle_groupings = [
            grouping
            for grouping in self.manifest.particle_groupings
            if grouping.id != grouping_id
        ]
        if removed.source_layer_id and self.manifest.active_particle_groupings.get(
            removed.source_layer_id
        ) == grouping_id:
            self.manifest.active_particle_groupings.pop(removed.source_layer_id, None)
            if removed.source_layer_id == self.current_layer_id:
                replacement = next(
                    (
                        grouping
                        for grouping in self.manifest.particle_groupings
                        if grouping.source_layer_id == self.current_layer_id
                    ),
                    None,
                )
                if replacement:
                    self.manifest.active_particle_groupings[
                        self.current_layer_id
                    ] = replacement.id
                    self.grouping_preview_definition = replacement.model_copy(deep=True)
                    self.grouping_preview = evaluate_particle_grouping(
                        self.particles, replacement
                    )
                    self.particles = self.grouping_preview.particles
                else:
                    self.grouping_preview = None
                    self.grouping_preview_definition = None
                    self.particles = [
                        particle.model_copy(update={"group": "Unclassified"})
                        for particle in self.particles
                    ]
                self.manifest.particle_records[self.current_layer_id] = self.particles
        self.manifest.edits.append(
            EditEvent(action="delete_particle_grouping", target_id=grouping_id)
        )
        self._refresh_grouping_panel()
        self._refresh_particles()
        self._render_visible_layers()
        self._set_dirty(True)

    def save_particle_grouping_overlay(self) -> Path | None:
        """Save the active grouping as a native-resolution Analysis-box overlay."""
        if self.manifest is None or self.current_layer_id is None:
            return None
        if self._pending_grouping_preview is not None:
            self.grouping_preview_timer.stop()
            self._apply_pending_grouping_preview()
        grouping = self.grouping_preview_definition
        layer = next(
            (
                item
                for item in self.manifest.layers
                if item.id == self.current_layer_id and item.kind == "instances"
            ),
            None,
        )
        if layer is None or grouping is None:
            QMessageBox.information(
                self,
                "Grouping overlay image",
                "Select an instance layer with an active particle grouping first.",
            )
            return None
        if layer.scope_roi_ids:
            roi = next(
                (
                    item
                    for item in self.manifest.rois
                    if layer.scope_roi_ids == [item.id]
                    and item.kind == ROIKind.ANALYSIS_BOX
                ),
                None,
            )
        else:
            roi = self._selected_analysis_box()
        if roi is None:
            QMessageBox.information(
                self,
                "Grouping overlay image",
                "Select the Analysis box associated with this particle layer first.",
            )
            return None
        suggested_directory = (
            self.current_path.parent if self.current_path is not None else Path.cwd()
        )
        suggested = suggested_directory / (
            f"particle_grouping_{grouping.id[:8]}_{roi.id[:8]}.png"
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save particle-grouping overlay",
            str(suggested),
            "PNG images (*.png)",
        )
        if not selected:
            return None
        target = Path(selected)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        try:
            image, bounds = load_region(
                self.current_path,
                self._roi_bounds(roi),
                color=True,
                scale_percent=100,
            )
            left, top, right, bottom = bounds
            labels = np.asarray(self.layer_masks[layer.id])[top:bottom, left:right]
            particles = self.manifest.particle_records.get(layer.id, self.particles)
            overlay = create_particle_group_overlay(
                image, labels, particles, grouping
            )
            if not cv2.imwrite(str(target), overlay):
                raise OSError(f"Could not write image: {target}")
        except Exception as error:
            QMessageBox.critical(self, "Grouping overlay save failed", str(error))
            return None
        self.statusBar().showMessage(
            f"Saved native-resolution grouping overlay to {target}", 5000
        )
        return target

    def save_color_plots(self, figure) -> list[Path]:
        """Choose a directory and save every color graph plus its Matplotlib Figure."""
        output = QFileDialog.getExistingDirectory(
            self, "Choose color-plot export directory"
        )
        if not output:
            return []
        if self.grouping_preview_definition is not None:
            context_id = self.grouping_preview_definition.id[:8]
        elif self.current_layer_id is not None:
            context_id = self.current_layer_id[:8]
        else:
            context_id = "particles"
        stem = f"color_plots_{context_id}"
        try:
            written = _save_color_plot_outputs(figure, output, stem)
        except Exception as error:
            QMessageBox.critical(self, "Color-plot save failed", str(error))
            return []
        self.statusBar().showMessage(
            f"Saved {len(written) - 1} plot images and one Matplotlib pickle to {output}",
            6000,
        )
        return written

    def plot_particles(self) -> object | None:
        if not self.particles:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PySide6.QtWidgets import QDialog

        dialog = QDialog(self)
        dialog.setWindowTitle("Particle size and circularity groups")
        dialog.resize(1250, 500)
        layout = QVBoxLayout(dialog)
        figure = Figure(figsize=(13, 4))
        canvas = FigureCanvasQTAgg(figure)
        axes = figure.subplots(1, 3)
        grouping = self.grouping_preview_definition
        result = self.grouping_preview
        if grouping and result:
            criteria = next(iter(grouping.groups), grouping.filter_criteria)
            buckets: list[tuple[str, str, list]] = []
            for group in grouping.groups:
                records = [
                    particle
                    for particle in result.particles
                    if result.assignments.get(particle.label) == group.id
                ]
                if group.enabled and records:
                    buckets.append((group.name, group.color, records))
            unclassified = [
                particle
                for particle in result.particles
                if particle.label in result.unclassified_labels
            ]
            filtered = [
                particle
                for particle in result.particles
                if particle.label in result.filtered_labels
            ]
            if grouping.show_unclassified and unclassified:
                buckets.append(("Unclassified", "#9e9e9e", unclassified))
            if grouping.show_filtered and filtered:
                buckets.append(("Filtered out", "#616161", filtered))
        else:
            criteria = None
            buckets = [("Particles", "#ffee58", self.particles)]
        if criteria is None:
            size_values = lambda particle: (
                particle.equivalent_radius_mm
                if self.manifest.calibration
                else particle.equivalent_radius_px
            )
            x_label = (
                "Equivalent radius (mm)"
                if self.manifest.calibration
                else "Equivalent radius (px)"
            )
        else:
            size_values = criteria.size_value
            metric_name = criteria.size_metric.value.replace("_", " ").title()
            unit = criteria.size_unit + ("²" if criteria.size_metric.value == "area" else "")
            x_label = f"{metric_name} ({unit})"
        histogram_values = []
        histogram_colors = []
        histogram_labels = []
        for name, color, records in buckets:
            values = [value for particle in records if (value := size_values(particle)) is not None]
            if not values:
                continue
            histogram_values.append(values)
            histogram_colors.append(color)
            histogram_labels.append(name)
            axes[1].scatter(
                values,
                [particle.circularity for particle in records if size_values(particle) is not None],
                s=10,
                alpha=0.7,
                color=color,
                label=name,
            )
        if histogram_values:
            axes[0].hist(
                histogram_values,
                bins="auto",
                color=histogram_colors,
                label=histogram_labels,
                alpha=0.75,
                stacked=True,
            )
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel("Count")
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("Circularity")
        analyzed_pixels, _ = self._grouping_statistics_context()
        pie_labels, pie_areas, pie_colors, pie_title = _particle_volume_pie_data(
            buckets, analyzed_pixels
        )
        if pie_areas:
            axes[2].pie(
                pie_areas,
                labels=pie_labels,
                colors=pie_colors,
                autopct=lambda percent: f"{percent:.2g}%",
                startangle=90,
            )
        else:
            axes[2].text(0.5, 0.5, "No particle area", ha="center", va="center")
        axes[2].set_title(pie_title)
        if len(buckets) > 1:
            axes[0].legend(fontsize="small")
            axes[1].legend(fontsize="small")
        figure.tight_layout()
        layout.addWidget(canvas)
        save_plots = QPushButton("Save plots as PNG + Matplotlib pickle…")
        save_plots.clicked.connect(lambda: self.save_color_plots(figure))
        layout.addWidget(save_plots)
        dialog.exec()
        return figure

    def show_selected_layer(self, item: QListWidgetItem) -> None:
        self.grouping_preview_timer.stop()
        self._pending_grouping_preview = None
        layer_id = item.data(Qt.ItemDataRole.UserRole)
        layer = next((value for value in self.manifest.layers if value.id == layer_id), None)
        if layer is None or layer_id not in self.layer_masks:
            return
        values = self.layer_masks[layer_id]
        self.layer_opacity.blockSignals(True)
        self.layer_opacity.setValue(layer.opacity)
        self.layer_opacity.blockSignals(False)
        if layer.kind == "domain":
            self.current_layer_id = None
            self.current_labels = None
            self.current_mask = None
            self.analysis_mask = None
            self.particles = []
        elif layer.kind == "multiclass":
            self.current_layer_id = layer.id
            self.current_labels = values.astype(np.int32)
            self.current_mask = None
            self.analysis_mask = None
            self.particles = []
        else:
            self.current_layer_id = layer.id
            self.current_labels = values.astype(np.int32)
            self.current_mask = (self.current_labels > 0).astype(np.uint8)
            self.particles = self.manifest.particle_records.get(layer.id, [])
            domain_layer = next(
                (
                    candidate
                    for candidate in self.manifest.layers
                    if candidate.kind == "domain"
                    and (
                        candidate.source_run_id == layer.source_run_id
                        or candidate.scope_roi_ids == layer.scope_roi_ids
                    )
                    and candidate.id in self.layer_masks
                ),
                None,
            )
            self.analysis_mask = (
                self.layer_masks[domain_layer.id].astype(bool)
                if domain_layer is not None
                else None
            )
        self.grouping_preview = None
        self.grouping_preview_definition = None
        active_grouping = self._active_particle_grouping()
        if active_grouping and self.particles:
            self.grouping_preview_definition = active_grouping.model_copy(deep=True)
            self.grouping_preview = evaluate_particle_grouping(
                self.particles, active_grouping
            )
        self._refresh_particles()
        self._refresh_grouping_panel()
        self._render_visible_layers()

    def _render_visible_layers(self) -> None:
        if self.manifest is None:
            self.canvas.set_mask_overlay(None)
            return
        classes = {item.id: item for item in self.manifest.classes}
        overlays = []
        for layer in self.manifest.layers:
            if not layer.visible or layer.id not in self.layer_masks:
                continue
            values = self.layer_masks[layer.id]
            if layer.kind == "multiclass":
                colors = {
                    value: self._class_rgb(classes[class_id].color)
                    for value, class_id in layer.class_value_map.items()
                    if class_id in classes
                }
            elif layer.kind == "domain":
                colors = (158, 158, 158)
            elif (
                layer.kind == "instances"
                and layer.id == self.current_layer_id
                and self.grouping_preview is not None
                and self.grouping_preview_definition is not None
            ):
                grouping = self.grouping_preview_definition
                group_colors = {
                    group.id: self._class_rgb(group.color)
                    for group in grouping.groups
                    if group.enabled
                }
                colors = {}
                for particle in self.grouping_preview.particles:
                    if particle.label in self.grouping_preview.filtered_labels:
                        if grouping.show_filtered:
                            colors[particle.label] = (97, 97, 97)
                        continue
                    group_id = self.grouping_preview.assignments.get(particle.label)
                    if group_id in group_colors:
                        colors[particle.label] = group_colors[group_id]
                    elif grouping.show_unclassified:
                        colors[particle.label] = (158, 158, 158)
            else:
                class_definition = classes.get(layer.class_id)
                colors = self._class_rgb(
                    class_definition.color if class_definition else "#ffee58"
                )
            overlays.append((values, colors, layer.opacity))
        self.canvas.set_composite_overlays(overlays)

    def _layer_visibility_changed(self, item: QListWidgetItem) -> None:
        if self.manifest is None:
            return
        layer_id = item.data(Qt.ItemDataRole.UserRole)
        layer = next((value for value in self.manifest.layers if value.id == layer_id), None)
        if layer is None:
            return
        layer.visible = item.checkState() == Qt.CheckState.Checked
        self._render_visible_layers()
        self._set_dirty(True)

    def _layer_opacity_changed(self, value: float) -> None:
        item = self.layers_list.currentItem()
        if item is None or self.manifest is None:
            return
        layer_id = item.data(Qt.ItemDataRole.UserRole)
        layer = next((entry for entry in self.manifest.layers if entry.id == layer_id), None)
        if layer:
            layer.opacity = value
            self._render_visible_layers()
            self._set_dirty(True)

    def _refresh_all(self) -> None:
        selected_binary_class = self.binary_class.currentData()
        selected_seed_class = self.seed_class.currentData()
        self.binary_class.blockSignals(True)
        self.seed_class.blockSignals(True)
        self.binary_class.clear()
        self.seed_class.clear()
        if self.manifest:
            available_classes = self._trainable_classes()
            for item in available_classes:
                self.binary_class.addItem(item.name, item.id)
            selected_index = self.binary_class.findData(selected_binary_class)
            if selected_index < 0:
                selected_index = next(
                    (
                        index
                        for index, item in enumerate(available_classes)
                        if item.preset == "particle"
                    ),
                    0,
                )
            self.binary_class.setCurrentIndex(selected_index)
            for item in available_classes:
                self.seed_class.addItem(item.name, item.id)
            seed_index = self.seed_class.findData(selected_seed_class)
            if seed_index < 0:
                seed_index = self.seed_class.findData(self.binary_class.currentData())
            self.seed_class.setCurrentIndex(max(0, seed_index))
        self.binary_class.blockSignals(False)
        self.seed_class.blockSignals(False)
        self._refresh_layers()
        self._refresh_rois()
        self._refresh_measurements()
        self._refresh_particles()
        self._refresh_grouping_panel()

    def _trainable_classes(self) -> list[ClassDefinition]:
        if self.manifest is None:
            return []
        return [item for item in self.manifest.classes if item.preset != "excluded"]

    def add_class(self) -> None:
        if self.manifest is None:
            return
        name, ok = QInputDialog.getText(self, "Add segmentation class", "Class name")
        if not ok or not name.strip():
            return
        color = QColorDialog.getColor(QColor("#ab47bc"), self, "Class color")
        if not color.isValid():
            return
        self.manifest.classes.append(ClassDefinition(name=name.strip(), color=color.name()))
        self._refresh_all()
        self._set_dirty(True)

    def rename_class(self) -> None:
        if self.manifest is None or self.seed_class.currentIndex() < 0:
            return
        class_id = self.seed_class.currentData()
        item = next((value for value in self.manifest.classes if value.id == class_id), None)
        if item is None:
            return
        name, ok = QInputDialog.getText(
            self, "Rename segmentation class", "Class name", text=item.name
        )
        if ok and name.strip():
            item.name = name.strip()
            self._refresh_all()
            self._set_dirty(True)

    def _refresh_layers(self) -> None:
        self.layers_list.blockSignals(True)
        self.layers_list.clear()
        if not self.manifest:
            self.layers_list.blockSignals(False)
            return
        for layer in self.manifest.layers:
            item = QListWidgetItem(layer.name)
            item.setData(Qt.ItemDataRole.UserRole, layer.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked
            )
            self.layers_list.addItem(item)
            if layer.id == self.current_layer_id:
                self.layers_list.setCurrentItem(item)
        self.layers_list.blockSignals(False)

    def _refresh_rois(self) -> None:
        selected_ids = {
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.rois_list.selectedItems()
        }
        self.rois_list.blockSignals(True)
        self.rois_list.clear()
        if self.manifest:
            for roi in self.manifest.rois:
                item = QListWidgetItem(f"{roi.name} [{roi.kind.value}]")
                item.setData(Qt.ItemDataRole.UserRole, roi.id)
                self.rois_list.addItem(item)
                item.setSelected(roi.id in selected_ids)
            self.canvas.set_roi_outlines(
                [
                    self._roi_polygon(roi)
                    for roi in self.manifest.rois
                ]
            )
        self.rois_list.blockSignals(False)

    def _roi_polygon(self, roi: ROI) -> list[QPointF]:
        points = [QPointF(point.x, point.y) for point in roi.points]
        if roi.shape == "rectangle" and len(points) >= 2:
            first, last = points[:2]
            return [
                QPointF(first.x(), first.y()),
                QPointF(last.x(), first.y()),
                QPointF(last.x(), last.y()),
                QPointF(first.x(), last.y()),
            ]
        return points

    def _roi_selected(self, item: QListWidgetItem) -> None:
        roi_id = item.data(Qt.ItemDataRole.UserRole)
        roi = next((value for value in self.manifest.rois if value.id == roi_id), None)
        if roi is None:
            return
        self.selected_roi_only.setChecked(True)
        if roi.recipe_id:
            recipe = next((value for value in self.manifest.recipes if value.id == roi.recipe_id), None)
            if recipe:
                self._set_recipe_controls(recipe)
        particle_candidates = [
            layer
            for layer in self.manifest.layers
            if layer.kind == "instances" and layer.scope_roi_ids == [roi.id]
        ]
        target_class_id = self.binary_class.currentData()
        matching_class = [
            layer for layer in particle_candidates if layer.class_id == target_class_id
        ]
        if matching_class:
            particle_candidates = matching_class
        candidates = particle_candidates or [
            layer
            for layer in self.manifest.layers
            if layer.kind == "multiclass" and layer.scope_roi_ids == [roi.id]
        ]
        if candidates:
            run_order = {run.id: index for index, run in enumerate(self.manifest.runs)}
            layer = max(
                candidates,
                key=lambda candidate: run_order.get(candidate.source_run_id, -1),
            )
            for row in range(self.layers_list.count()):
                layer_item = self.layers_list.item(row)
                if layer_item.data(Qt.ItemDataRole.UserRole) == layer.id:
                    self.layers_list.setCurrentItem(layer_item)
                    self.show_selected_layer(layer_item)
                    detail = (
                        "and its saved grouping scheme"
                        if layer.kind == "instances"
                        else "region-classification result"
                    )
                    self.statusBar().showMessage(
                        f"Selected {roi.name}: {layer.name} {detail}",
                        5000,
                    )
                    break
        else:
            self.current_layer_id = None
            self.current_labels = None
            self.current_mask = None
            self.analysis_mask = None
            self.particles = []
            self.grouping_preview = None
            self.grouping_preview_definition = None
            self._refresh_layers()
            self._refresh_particles()
            self._refresh_grouping_panel()
            self._render_visible_layers()
            self.statusBar().showMessage(
                f"Selected {roi.name}; run particles to create its ROI result",
                5000,
            )

    def _set_recipe_controls(self, recipe: SegmentationRecipe) -> None:
        target_index = self.binary_class.findData(recipe.target_class_id)
        if target_index >= 0:
            self.binary_class.setCurrentIndex(target_index)
        channel_index = self.channel.findData(recipe.channel)
        if channel_index >= 0:
            self.channel.setCurrentIndex(channel_index)
        self.segmentation_method.setCurrentIndex(
            self.segmentation_method.findData(recipe.segmentation_method)
        )
        self.method.setCurrentIndex(self.method.findData(recipe.threshold_method))
        self.polarity.setCurrentIndex(self.polarity.findData(recipe.polarity))
        self.illumination.setChecked(recipe.illumination_correction)
        self.split_touching.setChecked(recipe.split_touching)
        self.window_size.setValue(recipe.sauvola_window_px)
        self.sauvola_k.setValue(recipe.sauvola_k)
        self.gaussian_block.setValue(recipe.gaussian_block_px)
        self.gaussian_c.setValue(recipe.gaussian_c)
        self.manual_threshold_low.setValue(recipe.manual_threshold_low)
        self.manual_threshold_high.setValue(recipe.manual_threshold_high)
        self.gaussian_blur_sigma.setValue(recipe.gaussian_blur_sigma)
        self.fill_holes.setChecked(recipe.fill_holes)
        self.ball_radius.setValue(recipe.rolling_ball_radius_px)
        self.open_radius.setValue(recipe.open_radius_px)
        self.close_radius.setValue(recipe.close_radius_px)
        self.min_area.setValue(recipe.min_particle_area_px)
        self.max_area.setValue(recipe.max_particle_area_px or 0)
        self.watershed_distance.setValue(recipe.watershed_min_distance_px)
        self.morphological_input.setCurrentIndex(
            self.morphological_input.findData(recipe.morphological_input)
        )
        self.morphological_gradient.setCurrentIndex(
            self.morphological_gradient.findData(recipe.morphological_gradient)
        )
        self.morphological_gradient_radius.setValue(
            recipe.morphological_gradient_radius_px
        )
        self.morphological_tolerance.setValue(recipe.morphological_tolerance)
        self.morphological_connectivity.setCurrentIndex(
            self.morphological_connectivity.findData(recipe.morphological_connectivity)
        )
        self.morphological_calculate_dams.setChecked(recipe.morphological_calculate_dams)
        self.tile_size.setValue(recipe.tile_size_px)
        self._update_analysis_parameter_visibility()

    def _refresh_measurements(self) -> None:
        self.measurements_list.clear()
        if not self.manifest:
            return
        if self.manifest.calibration:
            calibration_item = QListWidgetItem(
                f"Calibration: {self.manifest.calibration.mm_per_pixel:.8g} mm/px"
            )
            calibration_item.setData(Qt.ItemDataRole.UserRole, "__calibration__")
            self.measurements_list.addItem(calibration_item)
        for item in self.manifest.measurements:
            value = (
                f"{item.length_mm:.6g} mm ({item.length_px:.3f} px)"
                if item.length_mm is not None
                else f"{item.length_px:.3f} px"
            )
            measurement_item = QListWidgetItem(f"{item.name}: {value}")
            measurement_item.setData(Qt.ItemDataRole.UserRole, item.id)
            self.measurements_list.addItem(measurement_item)
        lines: list[tuple[QPointF, QPointF, QColor]] = []
        calibration = self.manifest.calibration
        if calibration and calibration.reference_start and calibration.reference_end:
            lines.append(
                (
                    QPointF(calibration.reference_start.x, calibration.reference_start.y),
                    QPointF(calibration.reference_end.x, calibration.reference_end.y),
                    QColor("#00e5ff"),
                )
            )
        for measurement in self.manifest.measurements:
            lines.append(
                (
                    QPointF(measurement.start.x, measurement.start.y),
                    QPointF(measurement.end.x, measurement.end.y),
                    QColor("white"),
                )
            )
        self.canvas.set_annotations(lines)

    def _refresh_particles(self) -> None:
        records = (
            self.grouping_preview.particles
            if self.grouping_preview is not None
            else self.particles
        )
        grouping = self.grouping_preview_definition
        if grouping and self.grouping_preview and not grouping.show_filtered:
            records = [
                particle
                for particle in records
                if particle.label not in self.grouping_preview.filtered_labels
            ]
        calibrated = bool(self.manifest and self.manifest.calibration)
        criteria = (
            next(iter(grouping.groups), grouping.filter_criteria) if grouping else None
        )
        if criteria:
            metric_name = criteria.size_metric.value.replace("_", " ").title()
            squared = "²" if criteria.size_metric.value == "area" else ""
            self.particle_table.horizontalHeaderItem(1).setText(
                f"{metric_name} ({criteria.size_unit}{squared})"
            )
        else:
            self.particle_table.horizontalHeaderItem(1).setText(
                "Equivalent radius (mm)" if calibrated else "Equivalent radius (px)"
            )
        self.particle_table.setRowCount(len(records))
        for row, particle in enumerate(records):
            size = (
                criteria.size_value(particle)
                if criteria
                else (
                    particle.equivalent_radius_mm
                    if calibrated
                    else particle.equivalent_radius_px
                )
            )
            values = [
                str(particle.label),
                "" if size is None else f"{size:.6g}",
                f"{particle.circularity:.4f}",
                f"{particle.area_px:.0f}",
                f"{particle.solidity:.4f}",
                "yes" if particle.border_touching else "no",
                (
                    self.grouping_preview.group_names.get(particle.label, particle.group)
                    if self.grouping_preview
                    else particle.group
                ),
            ]
            for column, value in enumerate(values):
                self.particle_table.setItem(row, column, QTableWidgetItem(value))

    def _refresh_grouping_panel(self) -> None:
        if not self.manifest:
            self.grouping_panel.set_context(
                None, [], calibrated=False, groupings=[], active_id=None
            )
            return
        layer = next(
            (
                item
                for item in self.manifest.layers
                if item.id == self.current_layer_id and item.kind == "instances"
            ),
            None,
        )
        layer_id = layer.id if layer else None
        groupings = (
            [
                grouping
                for grouping in self.manifest.particle_groupings
                if grouping.source_layer_id == layer_id
            ]
            if layer_id
            else []
        )
        active_id = (
            self.manifest.active_particle_groupings.get(layer_id) if layer_id else None
        )
        self.grouping_panel.set_context(
            layer_id,
            self.particles if layer_id else [],
            calibrated=bool(self.manifest.calibration),
            groupings=groupings,
            active_id=active_id,
        )
        active = self._active_particle_grouping(layer_id)
        if active and self.particles:
            self._preview_particle_grouping(active)
        elif not layer_id:
            self.grouping_preview = None
            self.grouping_preview_definition = None
            self.grouping_panel.set_statistics({})

    def _show_summary(self, values: dict) -> None:
        lines = []
        for key, value in values.items():
            if isinstance(value, float):
                lines.append(f"{key.replace('_', ' ').title()}: {value:.6g}")
            else:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        self.summary.setText("\n".join(lines))

    @staticmethod
    def _class_rgb(color: str) -> tuple[int, int, int]:
        value = color.lstrip("#")
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved changes",
                "Close without saving the current project?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()


def QApplication_keyboard_control() -> bool:
    from PySide6.QtWidgets import QApplication

    return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
