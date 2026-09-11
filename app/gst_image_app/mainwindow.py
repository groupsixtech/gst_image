"""Main application window for interactive microstructure analysis."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QColor, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
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
    assign_particle_groups,
    build_analysis_mask,
    classify_regions,
    compute_project_fractions,
    create_measurement,
    measure_particles,
    segment_particles,
    suggest_specimen_mask,
)
from gst_image.analysis.groups import particle_group_summary
from gst_image.analysis.particles import merge_particle_labels, split_particle_by_line
from gst_image.analysis.preprocess import to_gray
from gst_image.export import export_analysis
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
    ClassDefinition,
    EditEvent,
    MorphologicalGradient,
    MorphologicalInput,
    ParticleGroup,
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
    return gray, domain, result


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
            ("Mask brush", "mask_brush"),
            ("Mask eraser", "mask_eraser"),
            ("Split", "split"),
        ]
        self.tool_actions: dict[str, QAction] = {}
        for label, name in tools:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=name: self.canvas.set_tool(value))
            toolbar.addAction(action)
            group.addAction(action)
            self.tool_actions[name] = action
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
        self.eyedropper_target.addItem("Manual threshold", "threshold")
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
        self.manual_threshold = QSpinBox()
        self.manual_threshold.setRange(0, 255)
        self.manual_threshold.setValue(128)
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
        self.manual_threshold_control, self.manual_threshold_slider = _slider_control(
            self.manual_threshold
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
        form.addRow("Manual threshold", self.manual_threshold_control)
        form.addRow("Morphological input", self.morphological_input)
        form.addRow("Gradient type", self.morphological_gradient)
        form.addRow("Morphological gradient radius (px)", self.morphological_gradient_control)
        form.addRow("Morphological tolerance", self.morphological_tolerance_control)
        form.addRow("Morphological connectivity", self.morphological_connectivity)
        form.addRow(self.morphological_calculate_dams)
        form.addRow("Gaussian pre-blur sigma (px)", self.gaussian_blur_control)
        form.addRow(self.illumination)
        form.addRow("Background radius (px)", self.ball_radius_control)
        form.addRow("Opening radius (px)", open_control)
        form.addRow("Closing radius (px)", close_control)
        form.addRow(self.fill_holes)
        form.addRow("Minimum area (px²)", min_area_control)
        form.addRow("Maximum area (px²)", max_area_control)
        form.addRow(self.split_touching)
        form.addRow("Watershed minimum distance (px)", self.watershed_distance_control)
        form.addRow("Tile size (px)", tile_control)
        form.addRow("Brush radius (px)", brush_control)
        layout.addLayout(form)
        layout.addWidget(self.selected_roi_only)
        self.restore_defaults_button = QPushButton("Restore analysis defaults")
        self.restore_defaults_button.clicked.connect(self.restore_analysis_defaults)
        layout.addWidget(self.restore_defaults_button)
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
            self.manual_threshold,
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
        class_buttons = QHBoxLayout()
        add_class = QPushButton("Add class")
        add_class.clicked.connect(self.add_class)
        rename_class = QPushButton("Rename class")
        rename_class.clicked.connect(self.rename_class)
        class_buttons.addWidget(add_class)
        class_buttons.addWidget(rename_class)
        layout.addLayout(class_buttons)
        self.train_regions_button = QPushButton("Train / update region classifier")
        self.train_regions_button.clicked.connect(self.train_regions)
        layout.addWidget(self.train_regions_button)
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
            ["Label", "Radius", "Circularity", "Area px²", "Solidity", "Border", "Group"]
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
        group_panel = QWidget()
        group_layout = QVBoxLayout(group_panel)
        self.groups_table = QTableWidget(0, 5)
        self.groups_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.groups_table.setHorizontalHeaderLabels(
            ["Name", "Min radius", "Max radius", "Min circularity", "Max circularity"]
        )
        group_layout.addWidget(self.groups_table)
        buttons = QHBoxLayout()
        add_group = QPushButton("Add")
        add_group.clicked.connect(self.add_group)
        apply_groups = QPushButton("Apply")
        apply_groups.clicked.connect(self.apply_groups)
        plot_groups = QPushButton("Plot")
        plot_groups.clicked.connect(self.plot_particles)
        delete_groups = QPushButton("Delete selected")
        delete_groups.clicked.connect(self.delete_selected_groups)
        buttons.addWidget(add_group)
        buttons.addWidget(apply_groups)
        buttons.addWidget(plot_groups)
        buttons.addWidget(delete_groups)
        group_layout.addLayout(buttons)
        tabs.addTab(group_panel, "Groups")
        dock.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _connect_canvas(self) -> None:
        self.canvas.line_finished.connect(self._line_finished)
        self.canvas.rectangle_finished.connect(self._rectangle_finished)
        self.canvas.polygon_finished.connect(self._polygon_finished)
        self.canvas.brush_stroke.connect(self._brush_stroke)
        self.canvas.scene_clicked.connect(self._select_particle)
        self.canvas.coordinates_changed.connect(
            lambda point: self.statusBar().showMessage(f"x={point.x():.1f}, y={point.y():.1f}")
        )

    def _set_dirty(self, dirty: bool = True) -> None:
        self.dirty = dirty
        suffix = " *" if dirty else ""
        self.setWindowTitle(f"GST Image — Weld Microstructure Analysis{suffix}")

    def _invalidate_preview_confirmation(self, *_args) -> None:
        self.confirmed_preview_recipe = None
        self.confirmed_preview_scope_ids = []
        if hasattr(self, "preview_status"):
            self.preview_status.setText("Preview parameters not confirmed")

    def _set_analysis_row_visible(self, field: QWidget, visible: bool) -> None:
        field.setVisible(visible)
        label = self.analysis_form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    def _update_analysis_parameter_visibility(self, *_args) -> None:
        """Show only controls used by the active thresholding configuration."""
        method = self.method.currentData()
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
            self.manual_threshold_control, method == ThresholdMethod.MANUAL
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
        self._set_analysis_row_visible(
            self.ball_radius_control, self.illumination.isChecked()
        )
        self.split_touching.setVisible(not morphological)
        self._set_analysis_row_visible(
            self.watershed_distance_control,
            not morphological and self.split_touching.isChecked(),
        )

    def restore_analysis_defaults(self) -> None:
        """Restore every analysis and preview control to its application default."""
        defaults = SegmentationRecipe()
        self._set_recipe_controls(defaults)
        if self.manifest is not None:
            particle_class = next(
                (item for item in self.manifest.classes if item.preset == "particle"), None
            )
            if particle_class is not None:
                self.binary_class.setCurrentIndex(
                    self.binary_class.findData(particle_class.id)
                )
        self.overview_resolution.setValue(25)
        self.roi_resolution.setValue(100)
        self.preview_resolution.setCurrentIndex(0)
        self.eyedropper_target.setCurrentIndex(0)
        self.selected_roi_only.setChecked(False)
        self.brush_radius.setValue(12)
        self._invalidate_preview_confirmation()
        self.statusBar().showMessage("Analysis parameters restored to defaults", 4000)

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

    def _roi_bounds(self, roi: ROI) -> tuple[int, int, int, int]:
        xs = [point.x for point in roi.points]
        ys = [point.y for point in roi.points]
        left = min(self.manifest.image_width - 1, max(0, int(np.floor(min(xs)))))
        top = min(self.manifest.image_height - 1, max(0, int(np.floor(min(ys)))))
        right = min(self.manifest.image_width, int(np.ceil(max(xs))) + 1)
        bottom = min(self.manifest.image_height, int(np.ceil(max(ys))) + 1)
        return left, top, max(left + 1, right), max(top + 1, bottom)

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
            manual_threshold=self.manual_threshold.value(),
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
        selected = self.rois_list.selectedItems() if self.selected_roi_only.isChecked() else []
        self.run_scope_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
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
            if roi and roi.recipe_id:
                recipe = recipe.model_copy(update={"id": roi.recipe_id, "name": f"{roi.name} recipe"})
                existing = next(
                    (index for index, item in enumerate(self.manifest.recipes) if item.id == roi.recipe_id),
                    None,
                )
                if existing is None:
                    self.manifest.recipes.append(recipe)
                else:
                    self.manifest.recipes[existing] = recipe
            else:
                self.manifest.recipes.append(recipe)
        else:
            self.manifest.recipes.append(recipe)
        self.run_recipe = recipe
        self.worker = FunctionWorker(
            _analyze_source,
            self.current_path,
            list(self.manifest.rois),
            self.run_scope_ids,
            recipe,
            self.manifest.calibration,
            with_callbacks=True,
        )
        self.worker.signals.progress.connect(self._progress)
        self.worker.signals.result.connect(self._particle_result)
        self.worker.signals.error.connect(self._worker_error)
        self.worker.signals.finished.connect(self._worker_finished)
        self.preview_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.preview_status.setText(
            f"Running original {self.manifest.image_width} × "
            f"{self.manifest.image_height} pixel source"
        )
        self.thread_pool.start(self.worker)

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
        gray, domain, result = payload
        result.summary.update(
            {
                "source_width_px": gray.shape[1],
                "source_height_px": gray.shape[0],
                "analysis_resolution": "original",
            }
        )
        self.gray = gray
        self.analysis_mask = domain
        self.current_mask = result.mask.astype(np.uint8)
        self.current_labels = result.labels
        self.particles = assign_particle_groups(result.particles, self.manifest.groups)
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
            suffix = "" if not self.run_scope_ids else f" — {len(self.manifest.layers) + 1}"
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
        self.manifest.particle_records[layer.id] = self.particles
        self._render_visible_layers()
        self._show_summary(result.summary)
        self._refresh_all()
        self._set_dirty(True)
        self.preview_status.setText(
            f"Full-resolution analysis complete ({gray.shape[1]} × {gray.shape[0]} px)"
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
        if tool == "seed":
            self._paint_training(points)
        elif tool in {"mask_brush", "mask_eraser"}:
            layer = next(
                (
                    item
                    for item in self.manifest.layers
                    if item.id == self.current_layer_id
                ),
                None,
            )
            if layer and layer.kind == "multiclass" and self.current_labels is not None:
                class_id = self.seed_class.currentData()
                value = next(
                    (
                        label
                        for label, mapped_class in layer.class_value_map.items()
                        if mapped_class == class_id
                    ),
                    0,
                )
                if value or tool == "mask_eraser":
                    self.undo_stack.push(
                        RegionPaintCommand(
                            self, points, value, tool == "mask_eraser"
                        )
                    )
            elif self.current_mask is not None:
                self.undo_stack.push(MaskLineCommand(self, points, tool == "mask_eraser"))

    def _paint_training(self, points: list[QPointF]) -> None:
        if self.training_labels is None or self.preview_bgr is None:
            return
        trainable = self._trainable_classes()
        class_index = self.seed_class.currentIndex()
        if class_index < 0:
            return
        label = class_index + 1
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
            )
        )
        colors = {index + 1: self._class_rgb(item.color) for index, item in enumerate(trainable)}
        self.canvas.set_label_overlay(self.training_labels, colors, 0.35)
        self._set_dirty(True)

    def _restore_training_strokes(self) -> None:
        if self.training_labels is None or self.preview_bgr is None:
            return
        trainable = self._trainable_classes()
        class_indexes = {item.id: index + 1 for index, item in enumerate(trainable)}
        sx = self.preview_bgr.shape[1] / self.manifest.image_width
        sy = self.preview_bgr.shape[0] / self.manifest.image_height
        for stroke in self.manifest.training_strokes:
            label = class_indexes.get(stroke.class_id)
            if label is None or not stroke.points:
                continue
            coords = [(round(p.x * sx), round(p.y * sy)) for p in stroke.points]
            radius = max(1, round(stroke.radius_px * (sx + sy) / 2))
            if len(coords) == 1:
                cv2.circle(self.training_labels, coords[0], radius, label, cv2.FILLED)
            else:
                cv2.line(self.training_labels, coords[0], coords[-1], label, 2 * radius + 1)

    def train_regions(self) -> None:
        if self.preview_bgr is None or self.training_labels is None or self.worker is not None:
            return
        recipe = self.manifest.region_recipes[-1] if self.manifest.region_recipes else RegionClassifierRecipe()
        worker = FunctionWorker(
            classify_regions,
            self.preview_bgr,
            self.training_labels.copy(),
            recipe,
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

    def _region_result(self, result) -> None:
        full_labels = cv2.resize(
            result.labels.astype(np.uint16),
            (self.manifest.image_width, self.manifest.image_height),
            interpolation=cv2.INTER_NEAREST,
        )
        layer = next((item for item in self.manifest.layers if item.kind == "multiclass"), None)
        trainable = self._trainable_classes()
        mapping = {index + 1: item.id for index, item in enumerate(trainable)}
        if layer is None:
            layer = SegmentationLayer(
                name="Weld regions", kind="multiclass", class_value_map=mapping
            )
            self.manifest.layers.append(layer)
        else:
            layer.class_value_map = mapping
        self.layer_masks[layer.id] = full_labels
        self.current_layer_id = layer.id
        self.current_labels = full_labels.astype(np.int32)
        self.current_mask = None
        self._render_visible_layers()
        self._show_summary(result.summary)
        self._refresh_layers()
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
        if target == "threshold":
            sampled_channel = to_gray(sample, self.channel.currentData())
            threshold = round(float(np.median(sampled_channel)))
            self.manual_threshold.setValue(threshold)
            self.method.setCurrentIndex(self.method.findData(ThresholdMethod.MANUAL))
            detail = f"manual threshold {threshold}"
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
        self.particles = assign_particle_groups(
            measure_particles(self.current_labels, self.manifest.calibration, self.analysis_mask),
            self.manifest.groups,
        )
        layer = next(
            (item for item in self.manifest.layers if item.id == self.current_layer_id), None
        )
        if layer:
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
        removed_ids = set(requested_ids)
        for run in self.manifest.runs:
            if run.id in run_ids:
                removed_ids.update(run.layer_ids)
        removed_ids.update(
            layer.id
            for layer in self.manifest.layers
            if layer.source_run_id in run_ids
        )
        self.manifest.layers = [
            layer for layer in self.manifest.layers if layer.id not in removed_ids
        ]
        self.manifest.runs = [run for run in self.manifest.runs if run.id not in run_ids]
        for layer_id in removed_ids:
            self.layer_masks.pop(layer_id, None)
            self.manifest.particle_records.pop(layer_id, None)
        if self.current_layer_id in removed_ids:
            self.current_layer_id = None
            self.current_mask = None
            self.current_labels = None
            self.analysis_mask = None
            self.particles = []
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
        rows = sorted(
            {index.row() for index in self.groups_table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row in rows:
            self.groups_table.removeRow(row)
        if rows:
            self.apply_groups()

    def add_group(self) -> None:
        row = self.groups_table.rowCount()
        self.groups_table.insertRow(row)
        defaults = [f"Group {row + 1}", "", "", "", ""]
        for column, value in enumerate(defaults):
            self.groups_table.setItem(row, column, QTableWidgetItem(value))

    def apply_groups(self) -> None:
        groups: list[ParticleGroup] = []
        unit = "mm" if self.manifest and self.manifest.calibration else "px"
        try:
            for row in range(self.groups_table.rowCount()):
                values = [
                    self.groups_table.item(row, column).text().strip()
                    if self.groups_table.item(row, column)
                    else ""
                    for column in range(5)
                ]
                groups.append(
                    ParticleGroup(
                        name=values[0] or f"Group {row + 1}",
                        radius_unit=unit,
                        radius_min=float(values[1]) if values[1] else None,
                        radius_max=float(values[2]) if values[2] else None,
                        circularity_min=float(values[3]) if values[3] else None,
                        circularity_max=float(values[4]) if values[4] else None,
                    )
                )
        except ValueError as error:
            QMessageBox.warning(self, "Invalid groups", str(error))
            return
        self.manifest.groups = groups
        self.particles = assign_particle_groups(self.particles, groups)
        layer = next(
            (item for item in self.manifest.layers if item.id == self.current_layer_id), None
        )
        if layer:
            self.manifest.particle_records[layer.id] = self.particles
        group_values = {group.name: index + 1 for index, group in enumerate(groups)}
        unclassified_value = len(groups) + 1
        label_groups = {
            particle.label: group_values.get(particle.group, unclassified_value)
            for particle in self.particles
        }
        colors = {
            index + 1: self._class_rgb(group.color) for index, group in enumerate(groups)
        }
        colors[unclassified_value] = (158, 158, 158)
        if self.current_labels is not None:
            self.canvas.set_particle_group_overlay(
                self.current_labels, label_groups, colors
            )
        self._refresh_particles()
        self.summary.setText(self.summary.text() + "\n\nGroups:\n" + json.dumps(particle_group_summary(self.particles), indent=2))
        self._set_dirty(True)

    def plot_particles(self) -> None:
        if not self.particles:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PySide6.QtWidgets import QDialog

        dialog = QDialog(self)
        dialog.setWindowTitle("Particle radius and circularity")
        dialog.resize(900, 500)
        layout = QVBoxLayout(dialog)
        figure = Figure(figsize=(9, 4))
        canvas = FigureCanvasQTAgg(figure)
        axes = figure.subplots(1, 2)
        radii = [
            p.equivalent_radius_mm if self.manifest.calibration else p.equivalent_radius_px
            for p in self.particles
        ]
        axes[0].hist(radii, bins="auto")
        axes[0].set_xlabel("Equivalent radius (mm)" if self.manifest.calibration else "Equivalent radius (px)")
        axes[0].set_ylabel("Count")
        axes[1].scatter(radii, [p.circularity for p in self.particles], s=8, alpha=0.65)
        axes[1].set_xlabel(axes[0].get_xlabel())
        axes[1].set_ylabel("Circularity")
        figure.tight_layout()
        layout.addWidget(canvas)
        dialog.exec()

    def show_selected_layer(self, item: QListWidgetItem) -> None:
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
        elif layer.kind == "multiclass":
            self.current_layer_id = layer.id
            self.current_labels = values.astype(np.int32)
            self.current_mask = None
        else:
            self.current_layer_id = layer.id
            self.current_labels = values.astype(np.int32)
            self.current_mask = (self.current_labels > 0).astype(np.uint8)
            self.particles = self.manifest.particle_records.get(layer.id, [])
            self._refresh_particles()
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
        self.binary_class.clear()
        self.seed_class.clear()
        if self.manifest:
            binary_classes = [
                item
                for item in self.manifest.classes
                if item.enabled and item.preset in {"particle", "phase", None}
            ]
            for item in binary_classes:
                self.binary_class.addItem(item.name, item.id)
            selected_index = self.binary_class.findData(selected_binary_class)
            if selected_index < 0:
                selected_index = next(
                    (
                        index
                        for index, item in enumerate(binary_classes)
                        if item.preset == "particle"
                    ),
                    0,
                )
            self.binary_class.setCurrentIndex(selected_index)
            for item in self._trainable_classes():
                self.seed_class.addItem(item.name, item.id)
        self._refresh_layers()
        self._refresh_rois()
        self._refresh_measurements()
        self._refresh_particles()
        self._refresh_groups()

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
        self.rois_list.clear()
        if self.manifest:
            for roi in self.manifest.rois:
                item = QListWidgetItem(f"{roi.name} [{roi.kind.value}]")
                item.setData(Qt.ItemDataRole.UserRole, roi.id)
                self.rois_list.addItem(item)
            self.canvas.set_roi_outlines(
                [
                    self._roi_polygon(roi)
                    for roi in self.manifest.rois
                ]
            )

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
        self.manual_threshold.setValue(round(recipe.manual_threshold))
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
        self.particle_table.setRowCount(len(self.particles))
        calibrated = bool(self.manifest and self.manifest.calibration)
        for row, particle in enumerate(self.particles):
            radius = particle.equivalent_radius_mm if calibrated else particle.equivalent_radius_px
            values = [
                str(particle.label),
                f"{radius:.6g}",
                f"{particle.circularity:.4f}",
                f"{particle.area_px:.0f}",
                f"{particle.solidity:.4f}",
                "yes" if particle.border_touching else "no",
                particle.group,
            ]
            for column, value in enumerate(values):
                self.particle_table.setItem(row, column, QTableWidgetItem(value))

    def _refresh_groups(self) -> None:
        if not self.manifest:
            return
        self.groups_table.setRowCount(0)
        for group in self.manifest.groups:
            row = self.groups_table.rowCount()
            self.groups_table.insertRow(row)
            values = [group.name, group.radius_min, group.radius_max, group.circularity_min, group.circularity_max]
            for column, value in enumerate(values):
                self.groups_table.setItem(row, column, QTableWidgetItem("" if value is None else str(value)))

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
