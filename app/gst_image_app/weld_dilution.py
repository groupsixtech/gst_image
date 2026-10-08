"""Guided weld dilution workspace; scientific operations live in the core package."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QObject, QPointF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QPainterPath, QPen, QUndoCommand
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi

from gst_image.analysis.masks import build_analysis_mask, suggest_specimen_mask
from gst_image.analysis.preprocess import to_gray
from gst_image.analysis.weld_dilution import (
    append_dilution_run,
    blurred_weld_preview,
    fit_surface_reference,
    line_bounds,
    measure_weld_dilution,
    scaled_weld_recipe,
    segment_weld_envelope,
)
from gst_image.dilution_export import export_dilution, plot_dilution, tie_line_table
from gst_image.image_io import load_image, load_preview, load_region, sha256_file
from gst_image.models import (
    EditEvent,
    Point,
    SegmentationLayer,
    WeldBlurStroke,
    WeldDilutionDraft,
    WeldDilutionRecipe,
    WeldSampling,
)
from gst_image_app.workers import FunctionWorker


def _spin(value, maximum=100000, *, integer=False, minimum=0):
    widget = QSpinBox() if integer else QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    if not integer:
        widget.setDecimals(6)
    widget.setValue(value)
    return widget


class TieLineTableModel(QAbstractTableModel):
    """Virtual rows keep large panoramas out of QTableWidget's per-cell allocation path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lines = []
        self.factor = 1
        self.unit = "px"

    def update(self, lines, factor, unit):
        self.beginResetModel()
        self.lines, self.factor, self.unit = lines, factor, unit
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.lines)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else 6

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        p = self.lines[index.row()]
        values = (
            p.position_px,
            p.depth_px,
            p.height_px,
            p.thickness_px,
            p.status,
            p.local_ratio_percent,
        )
        value = values[index.column()]
        if value is None:
            return ""
        if index.column() == 4:
            return value
        return f"{value * (self.factor if index.column() < 4 else 1):.6g}"

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
            names = ["Position", "Depth", "Height", "Thickness", "Status", "Local ratio (%)"]
            return names[section] + (f" ({self.unit})" if section < 4 else "")
        return None


class EnvelopeEdit(QUndoCommand):
    """Patch-based undo targets a draft layer, never the particle edit pipeline."""

    def __init__(self, controller, bounds, after, text, *, seeds=None):
        super().__init__(text)
        self.controller = controller
        self.project = controller.window.manifest
        self.draft = controller.draft
        self.layer_id = self.draft.envelope_layer_id
        self.bounds = bounds
        x0, y0, x1, y1 = bounds
        self.before = controller.envelope[y0:y1, x0:x1].copy()
        self.after = after.copy()
        self.seeds_before = list(self.draft.component_seeds)
        self.seeds_after = seeds
        self.manually_edited_before = self.draft.manually_edited

    def apply(self, values, undo=False):
        c = self.controller
        if c.window.manifest is not self.project or self.draft.envelope_layer_id != self.layer_id:
            return
        mask = c.window.layer_masks.get(self.layer_id)
        if mask is None:
            return
        x0, y0, x1, y1 = self.bounds
        mask[y0:y1, x0:x1] = values
        if self.seeds_after is not None:
            self.draft.component_seeds = self.seeds_before if undo else self.seeds_after
        self.draft.manually_edited = (
            self.manually_edited_before
            if undo
            else (self.manually_edited_before or self.seeds_after is None)
        )
        c.invalidate(draft=self.draft)
        self.project.edits.append(EditEvent(action="edit_weld_envelope", target_id=self.layer_id))
        c.render()

    def redo(self):
        self.apply(self.after)

    def undo(self):
        self.apply(self.before, undo=True)


class BlurStrokeEdit(QUndoCommand):
    def __init__(self, controller, strokes):
        super().__init__("Paint local weld blur")
        self.controller, self.project, self.draft = (
            controller, controller.window.manifest, controller.draft,
        )
        self.before = list(self.draft.recipe.blur_strokes)
        self.after = strokes

    def apply(self, strokes):
        c = self.controller
        if c.window.manifest is not self.project or self.draft not in self.project.dilution_drafts:
            return
        self.draft.recipe.blur_strokes = [s.model_copy(deep=True) for s in strokes]
        c.invalidate(segmentation=True, draft=self.draft)
        self.project.edits.append(EditEvent(action="edit_weld_blur", target_id=self.draft.id))
        if c.draft is self.draft and not c.history_run_id:
            c.preview_blur()

    def redo(self):
        self.apply(self.after)

    def undo(self):
        self.apply(self.before)


class WeldDilutionController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.draft = None
        self.history_run_id = None
        self.project = None
        self.candidates = None
        self.preview_result = None
        self.preview_image = None
        self.blurred_preview = None
        self.worker = None
        self.jobs = []
        self.epoch = 0
        self.loading = False
        self.selected_line = None
        self.annotation_items = []
        self._environment = None
        self._result_cache = object()
        self._analysis_was_visible = False
        self.original_dock = next(
            d for d in window.findChildren(QDockWidget) if d.windowTitle() == "Analysis"
        )
        self._build_controls()
        self._build_results()
        window.canvas.line_finished.connect(self.line_finished)
        window.canvas.scene_clicked.connect(self.scene_clicked)
        window.canvas.brush_stroke.connect(self.brush_stroke)
        window.canvas.polygon_finished.connect(self.polygon_finished)
        window.canvas.viewport_changed.connect(self.render_annotations)
        self.dock.visibilityChanged.connect(self.visibility_changed)
        analysis_menu = next(
            a.menu() for a in window.menuBar().actions() if a.text() == "&Analysis"
        )
        action = QAction("Weld Dilution", window)
        action.triggered.connect(self.show)
        analysis_menu.addAction(action)

    @property
    def active(self):
        return self.dock.isVisible() and self.draft is not None

    @property
    def envelope(self):
        if not self.draft:
            return None
        if self.history_run_id and self.run:
            return self.window.layer_masks.get(self.run.envelope_layer_id)
        layer_id = self.draft.envelope_layer_id or (
            self.run.envelope_layer_id if self.run else None
        )
        return self.window.layer_masks.get(layer_id)

    @property
    def domain(self):
        if not self.draft:
            return None
        if self.history_run_id and self.run:
            return self.window.layer_masks.get(self.run.domain_layer_id)
        layer_id = self.draft.domain_layer_id or (self.run.domain_layer_id if self.run else None)
        return self.window.layer_masks.get(layer_id)

    @property
    def run(self):
        if not self.draft or not self.window.manifest:
            return None
        run_id = self.history_run_id or self.draft.last_run_id
        return next((r for r in self.window.manifest.dilution_runs if r.id == run_id), None)

    @property
    def display_reference(self):
        return self.run.reference if self.history_run_id and self.run else self.draft.reference

    def show_history(self, run_id):
        self.cancel()
        self.history_run_id = run_id
        self.preview_result = self.preview_image = None
        run = self.run
        if run:
            snapshot = self.draft.model_copy(
                update={
                    "name": run.name,
                    "reference": run.reference,
                    "surface_points": run.reference.points,
                    "recipe": run.recipe,
                    "sampling": run.sampling,
                    "scope_roi_ids": run.scope_roi_ids,
                    "scope": "selected" if run.scope_roi_ids else "full",
                }
            )
            self.load_controls(snapshot)
        self.dock.widget().setEnabled(False)
        self.back_to_draft.show()
        self.status.setText("Viewing a saved snapshot. Return to current draft to edit.")
        self.refresh_results()
        self.render()

    def leave_history(self):
        self.history_run_id = None
        self.dock.widget().setEnabled(True)
        self.back_to_draft.hide()
        self.load_controls()
        self.refresh_results()
        self.render()

    def _button(self, layout, title, callback):
        button = QPushButton(title)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _tool(self, name):
        if name == "weld_blur_brush":
            self.window.canvas.set_brush_radius(self.blur_radius.value())
            self.stage.setCurrentIndex(1)
        if name in {"weld_brush", "weld_erase", "weld_polygon_add", "weld_polygon_remove"}:
            was_preview = self.preview_result is not None
            self.preview_result = self.preview_image = None
            if was_preview:
                self.window.show_overview()
            self.stage.setCurrentIndex(0)
            self.window.canvas.set_brush_radius(self.radius.value())
        elif name == "weld_select":
            self.stage.setCurrentIndex(2)
        self.window.canvas.set_tool(name)

    def _build_controls(self):
        self.dock = QDockWidget("Weld Dilution", self.window)
        self.dock.setObjectName("weldDilutionDock")
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.analysis_choice = QComboBox()
        self.analysis_choice.currentIndexChanged.connect(self.select_draft)
        layout.addWidget(self.analysis_choice)
        self._button(layout, "New named analysis", self.new_draft)
        form = QFormLayout()
        layout.addLayout(form)
        self.name = QLineEdit()
        form.addRow("Name", self.name)
        self.scope = QComboBox()
        self.scope.addItems(["Full panorama", "Selected ROI"])
        form.addRow("Scope", self.scope)
        self.setup_label = QLabel()
        self.setup_label.setWordWrap(True)
        layout.addWidget(self.setup_label)
        self._button(layout, "Calibrate", lambda: self.window._activate_tool("calibrate"))
        self._button(layout, "Draw surface line", lambda: self._tool("weld_line"))
        self._button(layout, "Add surface points", lambda: self._tool("weld_points"))
        self.points = QTableWidget(0, 2)
        self.points.setHorizontalHeaderLabels(["Source X", "Source Y"])
        self.points.setMaximumHeight(140)
        layout.addWidget(self.points)
        self._button(layout, "Fit / apply edited points", self.fit_points)
        self._button(layout, "Delete selected surface points", self.delete_points)
        self.flip = QCheckBox("Flip substrate side")
        layout.addWidget(self.flip)
        self.reference_label = QLabel("Draw a line along the visible unmelted surface")
        self.reference_label.setWordWrap(True)
        layout.addWidget(self.reference_label)
        layout.addWidget(QLabel("Gaussian blur — inspect smoothing first"))
        form = QFormLayout()
        layout.addLayout(form)
        self.channel = QComboBox()
        self.channel.addItems(["gray", "blue", "green", "red", "lab_l", "lab_a", "lab_b"])
        self.sigma = _spin(10)
        form.addRow("Channel", self.channel)
        form.addRow("Blur sigma (source px)", self.sigma)
        self.blur_button = self._button(layout, "Preview blur", self.preview_blur)
        self._button(layout, "Show original", lambda: self.stage.setCurrentIndex(0))
        self._button(layout, "Inspect native detail (click image)",
                     lambda: self._tool("weld_blur_detail"))
        self._button(layout, "Return to panorama", self.window.show_overview)
        self.blur_info = QLabel("Preview blur, then inspect native detail to judge smoothing.")
        self.blur_info.setWordWrap(True)
        layout.addWidget(self.blur_info)
        brush_form = QFormLayout()
        layout.addLayout(brush_form)
        self.blur_radius = _spin(50, integer=True, minimum=1)
        self.brush_sigma = _spin(50, minimum=0.01)
        self.brush_sigma.setToolTip(
            "Use a sigma greater than the global blur to add local smoothing. "
            "Overlapping strokes use the last painted strength; blur does not accumulate."
        )
        brush_form.addRow("Blur brush radius (source px)", self.blur_radius)
        brush_form.addRow("Local sigma (source px)", self.brush_sigma)
        self.blur_radius.valueChanged.connect(
            lambda value: self.window.canvas.set_brush_radius(value)
            if self.window.canvas.current_tool == "weld_blur_brush" else None
        )
        self._button(layout, "Paint local blur", lambda: self._tool("weld_blur_brush"))
        self._button(layout, "Clear local blur strokes", self.clear_blur_strokes)
        layout.addWidget(QLabel("Adaptive segmentation — apply after checking blur"))
        form = QFormLayout()
        layout.addLayout(form)
        self.polarity = QComboBox()
        self.polarity.addItems(["dark", "bright"])
        self.method = QComboBox()
        self.method.addItems(["adaptive_gaussian", "sauvola"])
        self.window_size = _spin(201, integer=True, minimum=3)
        self.c = _spin(2, 255, minimum=-255)
        self.k = _spin(0.2, 1, minimum=-1)
        self.closing = _spin(3, integer=True)
        self.opening = _spin(0, integer=True)
        for label, widget in (
            ("Polarity", self.polarity),
            ("Adaptive method", self.method),
            ("Window (source px)", self.window_size),
            ("Gaussian C", self.c),
            ("Sauvola k", self.k),
            ("Closing radius (px)", self.closing),
            ("Opening radius (px)", self.opening),
        ):
            form.addRow(label, widget)
        self._button(layout, "Preview segmentation", lambda: self.segment(preview=True))
        self._button(layout, "Generate full-resolution envelope", self.segment)
        self.stage = QComboBox()
        self.stage.addItems(["Original + envelope", "Blurred preview", "Threshold candidates"])
        layout.addWidget(self.stage)
        self._button(layout, "Select / deselect weld components", lambda: self._tool("weld_select"))
        self._button(layout, "Brush envelope", lambda: self._tool("weld_brush"))
        self._button(layout, "Erase envelope", lambda: self._tool("weld_erase"))
        self._button(layout, "Polygon add", lambda: self._tool("weld_polygon_add"))
        self._button(layout, "Polygon remove", lambda: self._tool("weld_polygon_remove"))
        form = QFormLayout()
        layout.addLayout(form)
        self.radius = _spin(12, integer=True, minimum=1)
        form.addRow("Envelope brush radius (px)", self.radius)
        self.radius.valueChanged.connect(self.window.canvas.set_brush_radius)
        layout.addWidget(QLabel("Measurements and reporting"))
        form = QFormLayout()
        layout.addLayout(form)
        self.spacing = _spin(1, minimum=0.000001)
        self.spacing_unit = QComboBox()
        self.spacing_unit.addItems(["px", "mm", "µm"])
        self.display_unit = QComboBox()
        self.display_unit.addItems(["mm", "µm", "px"])
        self.length = _spin(0)
        self.length.setSpecialValueText("No volume estimate")
        self.bins = _spin(30, 500, integer=True, minimum=1)
        for label, widget in (
            ("Tie-line spacing", self.spacing),
            ("Spacing unit", self.spacing_unit),
            ("Display unit", self.display_unit),
            ("Weld length (mm)", self.length),
            ("Histogram bins", self.bins),
        ):
            form.addRow(label, widget)
        self.overlay = QCheckBox("Show envelope and tie-lines")
        self.overlay.setChecked(True)
        layout.addWidget(self.overlay)
        self.opacity = _spin(0.4, 1)
        form.addRow("Overlay opacity", self.opacity)
        self._button(layout, "Calculate", self.calculate)
        self.confirm_button = self._button(layout, "Confirm reviewed", self.confirm)
        self._button(layout, "Save project", self.window.save_project_dialog)
        self._button(layout, "Export dilution", self.export)
        self._button(layout, "Cancel processing", self.cancel)
        self.status = QLabel("Open a panorama to begin")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        note = QLabel(
            "Review the crown and fusion boundary on the original image. "
            "Exclude HAZ. Internal holes are filled. Volume assumes a constant "
            "cross-section along the entered length."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        self.dock.setWidget(scroll)
        self.window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.dock.hide()
        self.name.editingFinished.connect(self.settings_changed)
        self.scope.currentIndexChanged.connect(lambda: self.settings_changed(segmentation=True))
        for widget in (self.channel, self.polarity, self.method):
            widget.currentIndexChanged.connect(lambda: self.settings_changed(segmentation=True))
        for widget in (self.sigma, self.window_size, self.c, self.k, self.closing, self.opening):
            widget.valueChanged.connect(lambda: self.settings_changed(segmentation=True))
        for widget in (self.spacing, self.length):
            widget.valueChanged.connect(self.settings_changed)
        self.spacing_unit.currentIndexChanged.connect(self.settings_changed)
        self.flip.toggled.connect(self.fit_points)
        self.points.cellChanged.connect(self.fit_points)
        self.display_unit.currentIndexChanged.connect(self.plot_settings_changed)
        self.bins.valueChanged.connect(self.plot_settings_changed)
        self.stage.currentIndexChanged.connect(self.render)
        self.overlay.toggled.connect(self.render)
        self.opacity.valueChanged.connect(self.render)

    def _build_results(self):
        self.results = QWidget()
        layout = QVBoxLayout(self.results)
        self.back_to_draft = self._button(layout, "Return to current draft", self.leave_history)
        self.back_to_draft.hide()
        self.summary_label = QLabel("No dilution result")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.result_tabs = QTabWidget()
        layout.addWidget(self.result_tabs)
        self.table = QTableView()
        self.tie_model = TieLineTableModel(self.table)
        self.table.setModel(self.tie_model)
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.selectionModel().selectionChanged.connect(self.highlight_row)
        self.result_tabs.addTab(self.table, "Tie-lines")
        self.statistics = QTableWidget()
        self.statistics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_tabs.addTab(self.statistics, "Statistics")
        self.figures = [Figure(figsize=(5, 3)), Figure(figsize=(5, 3))]
        self.plots = [FigureCanvasQTAgg(f) for f in self.figures]
        self.result_tabs.addTab(self.plots[0], "Depth / height profile")
        self.result_tabs.addTab(self.plots[1], "Histograms")
        self.plots[0].mpl_connect("pick_event", self.plot_picked)
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        self._button(buttons, "Copy tie-lines", self.copy_table)
        self._button(buttons, "Copy summary / statistics", self.copy_summary)
        self.window.results_tabs.addTab(self.results, "Dilution")

    def show(self):
        self.leave_history()
        self.observe()
        if not self.window.manifest:
            self.status.setText("Open a panorama first")
        elif not self.draft:
            self.new_draft()
        if not self.dock.isVisible():
            self._analysis_was_visible = self.original_dock.isVisible()
            self.original_dock.hide()
        self.dock.show()
        self.dock.raise_()
        self.window.resizeDocks([self.dock], [360], Qt.Orientation.Horizontal)
        QTimer.singleShot(0, self.window.canvas, self.fit_canvas)
        self.window.results_tabs.setCurrentWidget(self.results)
        self.render()

    def fit_canvas(self):
        canvas = self.window.canvas
        canvas.fitInView(canvas.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.render_annotations()

    def visibility_changed(self, visible):
        if not visible:
            self.window.canvas.set_base_preview(None)
            if self.window.canvas.current_tool.startswith("weld_"):
                self.window.canvas.set_tool("pan")
            self.clear_annotations()
            if self._analysis_was_visible:
                self.original_dock.show()
                self._analysis_was_visible = False
            self.window._render_visible_layers()
        elif self.draft:
            self.render()

    def _environment_signature(self):
        m = self.window.manifest
        return (
            m.source_revision,
            m.calibration.model_dump_json() if m.calibration else None,
            json.dumps([r.model_dump(mode="json") for r in m.rois], sort_keys=True),
        )

    def observe(self):
        """Called on project/ROI/calibration changes, without touching particle state."""
        m = self.window.manifest
        if m is not self.project:
            self.cancel()
            self.project = m
            self.draft = None
            self.history_run_id = None
            self.dock.widget().setEnabled(True)
            self.back_to_draft.hide()
            self.candidates = self.preview_result = self.preview_image = None
            self.blurred_preview = None
            self.clear_annotations()
            self._environment = self._environment_signature() if m else None
            self.refresh_choices()
            if m and m.dilution_drafts:
                self.draft = m.dilution_drafts[0]
                self.load_controls()
            self.refresh_results()
            self.render()
        elif m:
            signature = self._environment_signature()
            if signature != self._environment:
                self.cancel()
                geometry_changed = (
                    signature[0] != self._environment[0] or signature[2] != self._environment[2]
                )
                self._environment = signature
                for draft in m.dilution_drafts:
                    self.invalidate(segmentation=geometry_changed, draft=draft, dirty=False)
                self.load_setup_label()
        if m and self.draft and self.draft not in m.dilution_drafts:
            self.cancel()
            self.draft = None
            self.candidates = None
            self.refresh_choices()
        if m and self.draft:
            ids = {r.id for r in m.rois}
            self.draft.scope_roi_ids = [r for r in self.draft.scope_roi_ids if r in ids]
        if self.history_run_id and (
            not m or not any(r.id == self.history_run_id for r in m.dilution_runs)
        ):
            self.leave_history()
        self.refresh_results()

    def refresh_choices(self):
        self.analysis_choice.blockSignals(True)
        self.analysis_choice.clear()
        if self.window.manifest:
            for draft in self.window.manifest.dilution_drafts:
                self.analysis_choice.addItem(draft.name, draft.id)
            if self.draft:
                self.analysis_choice.setCurrentIndex(self.analysis_choice.findData(self.draft.id))
        self.analysis_choice.blockSignals(False)

    def new_draft(self):
        if not self.window.manifest:
            return
        self.leave_history()
        self.cancel()
        self.draft = WeldDilutionDraft(
            name=f"Weld dilution {len(self.window.manifest.dilution_drafts) + 1}"
        )
        self.window.manifest.dilution_drafts.append(self.draft)
        self.candidates = self.preview_result = self.preview_image = None
        self.blurred_preview = None
        self.refresh_choices()
        self.load_controls()
        self.window._set_dirty()
        self.refresh_results()
        self.render()

    def select_draft(self):
        if self.loading or not self.window.manifest:
            return
        self.leave_history()
        self.cancel()
        self.draft = next(
            (
                d
                for d in self.window.manifest.dilution_drafts
                if d.id == self.analysis_choice.currentData()
            ),
            None,
        )
        self.candidates = self.preview_result = self.preview_image = None
        self.blurred_preview = None
        if self.draft:
            self.load_controls()
            self.refresh_results()
            self.render()

    def load_setup_label(self):
        m = self.window.manifest
        if m:
            calibration = (
                self.run.calibration if self.history_run_id and self.run else m.calibration
            )
            scale = f"{calibration.mm_per_pixel:.6g} mm/px" if calibration else "Uncalibrated"
            self.setup_label.setText(
                f"{m.image_width} × {m.image_height} source pixels; {scale}. "
                "Exclusion ROIs apply to both areas and profiles."
            )

    def load_controls(self, draft=None):
        d = draft or self.draft
        if not d:
            return
        self.loading = True
        r, s = d.recipe, d.sampling
        self.name.setText(d.name)
        self.scope.setCurrentIndex(0 if d.scope == "full" else 1)
        for widget, value in (
            (self.channel, r.channel),
            (self.polarity, r.polarity),
            (self.method, r.threshold_method),
            (self.spacing_unit, s.spacing_unit),
            (self.display_unit, s.display_unit),
        ):
            widget.setCurrentText(value)
        for widget, value in (
            (self.sigma, r.gaussian_sigma_px),
            (self.window_size, r.window_px),
            (self.c, r.adaptive_c),
            (self.k, r.sauvola_k),
            (self.closing, r.close_radius_px),
            (self.opening, r.open_radius_px),
            (self.spacing, s.spacing),
            (self.length, s.weld_length_mm or 0),
            (self.bins, s.histogram_bins),
        ):
            widget.setValue(value)
        self.flip.setChecked(d.reference.flipped if d.reference else d.substrate_flipped)
        self.set_points(d.surface_points or (d.reference.points if d.reference else []))
        self.loading = False
        self.load_setup_label()
        self.update_reference_label()

    def set_points(self, points):
        self.points.blockSignals(True)
        self.points.setRowCount(len(points))
        for row, p in enumerate(points):
            for col, value in enumerate((p.x, p.y)):
                self.points.setItem(row, col, QTableWidgetItem(f"{value:.10g}"))
        self.points.blockSignals(False)

    def settings_changed(self, *_args, segmentation=False):
        if self.loading or not self.draft or self.history_run_id:
            return
        d = self.draft
        blur_changed = (
            d.recipe.channel != self.channel.currentText()
            or d.recipe.gaussian_sigma_px != self.sigma.value()
        )
        d.name = self.name.text().strip() or "Weld dilution"
        d.recipe = WeldDilutionRecipe(
            blur_strokes=d.recipe.blur_strokes,
            channel=self.channel.currentText(),
            polarity=self.polarity.currentText(),
            threshold_method=self.method.currentText(),
            gaussian_sigma_px=self.sigma.value(),
            window_px=self.window_size.value(),
            adaptive_c=self.c.value(),
            sauvola_k=self.k.value(),
            close_radius_px=self.closing.value(),
            open_radius_px=self.opening.value(),
        )
        d.sampling = WeldSampling(
            spacing=self.spacing.value(),
            spacing_unit=self.spacing_unit.currentText(),
            display_unit=self.display_unit.currentText(),
            histogram_bins=self.bins.value(),
            weld_length_mm=self.length.value() or None,
        )
        d.scope = "full" if self.scope.currentIndex() == 0 else "selected"
        if d.scope == "full":
            d.scope_roi_ids = []
        self.refresh_choices()
        self.invalidate(segmentation=segmentation, blur=blur_changed)
        self.render()

    def plot_settings_changed(self):
        if self.loading or not self.draft:
            return
        self.draft.sampling.display_unit = self.display_unit.currentText()
        self.draft.sampling.histogram_bins = self.bins.value()
        if self.run:
            self.run.sampling.display_unit = self.display_unit.currentText()
            self.run.sampling.histogram_bins = self.bins.value()
        self.window._set_dirty()
        self.refresh_results()

    def invalidate(self, *, segmentation=False, draft=None, dirty=True, blur=True):
        d = draft or self.draft
        if not d:
            return
        self.cancel()
        if segmentation:
            d.segmentation_stale = d.envelope_layer_id is not None
            if blur:
                self.blurred_preview = None
                self.blur_info.setText("Blur settings changed — click Preview blur to update.")
        if self.window.manifest:
            run = next(
                (r for r in self.window.manifest.dilution_runs if r.id == d.last_run_id), None
            )
            if run:
                run.outdated = True
                run.review_status = "pending"
                run.reviewed_at = None
                for layer in self.window.manifest.layers:
                    if layer.id in run.layer_ids:
                        layer.review_status = "pending"
        self.status.setText(
            "Envelope settings changed — regenerate"
            if d.segmentation_stale
            else "Inputs changed — calculate to update results"
        )
        if dirty:
            self.window._set_dirty()
        self.refresh_results()

    def fit_points(self, *_args):
        if self.loading or not self.draft or self.history_run_id:
            return
        try:
            points = [
                (float(self.points.item(row, 0).text()), float(self.points.item(row, 1).text()))
                for row in range(self.points.rowCount())
            ]
            if not np.isfinite(points).all():
                raise ValueError("Surface coordinates must be finite")
            self.draft.surface_points = [Point(x=x, y=y) for x, y in points]
            self.draft.substrate_flipped = self.flip.isChecked()
            self.draft.reference = fit_surface_reference(points, flipped=self.flip.isChecked())
        except (ValueError, AttributeError) as error:
            self.draft.reference = None
            self.reference_label.setText(str(error))
            self.invalidate()
            self.render()
            return
        self.invalidate()
        self.update_reference_label()
        self.render()

    def update_reference_label(self):
        if self.draft and self.display_reference:
            r = self.display_reference
            self.reference_label.setText(
                f"Fit RMS {r.rms_residual_px:.4g} px; reference span {r.span_px:.5g} px. "
                "Arrow points into the substrate."
            )

    def delete_points(self):
        self.points.blockSignals(True)
        for row in sorted({i.row() for i in self.points.selectedIndexes()}, reverse=True):
            self.points.removeRow(row)
        self.points.blockSignals(False)
        self.fit_points()

    def line_finished(self, tool, start, end):
        if tool == "weld_line" and self.draft and not self.history_run_id:
            self.set_points([Point(x=start.x(), y=start.y()), Point(x=end.x(), y=end.y())])
            self.fit_points()

    def scene_clicked(self, point):
        if not self.active or self.history_run_id:
            return
        tool = self.window.canvas.current_tool
        if tool == "weld_blur_detail":
            self.inspect_blur_detail(point)
            return
        if tool == "weld_points":
            self.points.blockSignals(True)
            row = self.points.rowCount()
            self.points.insertRow(row)
            for col, v in enumerate((point.x(), point.y())):
                self.points.setItem(row, col, QTableWidgetItem(str(v)))
            self.points.blockSignals(False)
            self.fit_points()
        elif tool == "weld_select":
            if (
                self.candidates is None
                or self.envelope is None
                or self.draft.segmentation_stale
                or self.preview_result is not None
            ):
                self.status.setText(
                    "Generate full-resolution candidates before selecting components"
                )
                return
            x, y = round(point.x()), round(point.y())
            if not (0 <= y < self.candidates.shape[0] and 0 <= x < self.candidates.shape[1]):
                return
            label = int(self.candidates[y, x])
            if not label:
                return
            seeds = self.draft.component_seeds
            existing = [p for p in seeds if self.candidates[round(p.y), round(p.x)] == label]
            new_seeds = (
                [p for p in seeds if p not in existing] if existing else [*seeds, Point(x=x, y=y)]
            )
            selected = self.candidates == label
            ys = np.flatnonzero(np.any(selected, axis=1))
            xs = np.flatnonzero(np.any(selected, axis=0))
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            after = self.envelope[y0:y1, x0:x1].copy()
            after[selected[y0:y1, x0:x1]] = 0 if existing else 1
            self.window.undo_stack.push(
                EnvelopeEdit(
                    self,
                    (x0, y0, x1, y1),
                    after,
                    "Select weld component",
                    seeds=new_seeds,
                )
            )

    def _start_job(self, function, done):
        self.cancel()
        generation, project = self.epoch, self.window.manifest
        source_revision, source_hash = project.source_revision, project.source_sha256
        worker = FunctionWorker(function, with_callbacks=True)
        self.worker = worker
        self.jobs.append(worker)
        worker.signals.progress.connect(
            lambda value, message: (
                self.status.setText(f"{value:.0%} {message}") if self.worker is worker else None
            )
        )

        def complete(value):
            if (
                generation == self.epoch
                and project is self.window.manifest
                and project.source_revision == source_revision
                and project.source_sha256 == source_hash
                and not worker.cancel_event.is_set()
            ):
                done(value)

        def finish():
            if worker in self.jobs:
                self.jobs.remove(worker)
            if self.worker is worker:
                self.worker = None

        worker.signals.result.connect(complete)
        def error(message):
            if generation == self.epoch:
                self.status.setText(message)
                self.blur_info.setText(message)

        worker.signals.error.connect(error)
        worker.signals.finished.connect(finish)
        self.window.thread_pool.start(worker)

    def cancel(self):
        self.epoch += 1
        if self.worker:
            self.worker.cancel()
            self.worker = None

    def preview_blur(self, *_args):
        if not self.draft or not self.window.current_path or self.history_run_id:
            return
        recipe = self.draft.recipe.model_copy(deep=True)
        path = self.window.current_path
        source_hash = self.window.manifest.source_sha256

        def work(*, progress, cancelled):
            if cancelled():
                raise InterruptedError("Weld dilution cancelled")
            if sha256_file(path) != source_hash:
                raise ValueError("Source image changed; reopen or relink it as a new revision")
            return blurred_weld_preview(
                load_image(path, color=True), recipe, progress=progress, cancelled=cancelled,
                max_size=None,
            )

        def done(blurred):
            self.blurred_preview = blurred
            if self.window.canvas.current_tool != "weld_blur_brush":
                self.window.canvas.set_tool("pan")
            self.stage.setCurrentIndex(1)
            self.render()
            self.status.setText(
                "Blur-only preview — inspect smoothing, then Preview segmentation. "
                "Use Inspect native detail to compare at full resolution."
            )
            screen_sigma = recipe.gaussian_sigma_px * self.window.canvas.transform().m11()
            maximum_sigma = max(
                [recipe.gaussian_sigma_px] + [s.sigma_px for s in recipe.blur_strokes]
            )
            self.blur_info.setText(
                f"Blur displayed: sigma {recipe.gaussian_sigma_px:g} source px "
                f"(about {screen_sigma:.2f} screen px at this zoom); "
                f"{len(recipe.blur_strokes)} local strokes. "
                f"Maximum local sigma {maximum_sigma:g} px. "
                "Small screen values can look unchanged. Inspect native detail or increase sigma."
            )

        self.status.setText("Generating blur-only preview")
        self.blur_info.setText("Generating native blur preview…")
        self._start_job(work, done)

    def inspect_blur_detail(self, point):
        path = self.window.current_path
        x, y = round(point.x()), round(point.y())
        bounds = (x - 256, y - 256, x + 256, y + 256)

        def work(*, progress, cancelled):
            return load_region(path, bounds, color=True)

        def done(value):
            image, rect = value
            self.window._set_canvas_image(image, rect)
            self.window.canvas.resetTransform()
            self.window.canvas.centerOn(x, y)
            self.window.canvas.set_tool("pan")
            self.blur_info.setText(
                "Native detail: 1 source pixel per screen pixel. "
                "Compare Original + envelope and Blurred preview, or paint local blur."
            )
            self.render()

        self._start_job(work, done)

    def clear_blur_strokes(self):
        if self.draft and not self.history_run_id and self.draft.recipe.blur_strokes:
            self.window.undo_stack.push(BlurStrokeEdit(self, []))

    def segment(self, *_args, preview=False):
        if not self.draft or not self.window.current_path or self.history_run_id:
            return
        if self.draft.scope == "selected":
            roi = self.window._selected_full_resolution_roi()
            if roi and self.draft.scope_roi_ids != [roi.id]:
                self.draft.scope_roi_ids = [roi.id]
                self.invalidate(segmentation=True)
            if not self.draft.scope_roi_ids:
                self.status.setText("Select an Include or Analysis ROI first")
                return
        draft = self.draft.model_copy(deep=True)
        rois = [r.model_copy(deep=True) for r in self.window.manifest.rois]
        path = self.window.current_path
        source_hash = self.window.manifest.source_sha256
        width, height = self.window.manifest.image_width, self.window.manifest.image_height

        def work(*, progress, cancelled):
            if cancelled():
                raise InterruptedError("Weld dilution cancelled")
            if sha256_file(path) != source_hash:
                raise ValueError("Source image changed; reopen or relink it as a new revision")
            if preview:
                image, _ = load_preview(path, color=True)
                sx, sy = image.shape[1] / width, image.shape[0] / height
                for roi in rois:
                    roi.points = [Point(x=p.x * sx, y=p.y * sy) for p in roi.points]
                recipe = scaled_weld_recipe(draft.recipe, min(sx, sy))
            else:
                image = load_image(path, color=True)
                recipe = draft.recipe
            domain = build_analysis_mask(
                image.shape[:2],
                rois,
                suggest_specimen_mask(to_gray(image)),
                include_ids=set(draft.scope_roi_ids),
            )
            result = segment_weld_envelope(
                image,
                domain,
                recipe,
                progress=progress,
                cancelled=cancelled,
            )
            return result, domain, image if preview else None

        def done(values):
            result, domain, image = values
            if preview:
                self.preview_result, self.preview_image = result, image
                self.window.canvas.set_image(image, (width, height))
                self.stage.setCurrentIndex(2)
                self.status.setText(
                    "PROVISIONAL preview — generate full-resolution candidates next"
                )
            else:
                self.invalidate()
                self.candidates = result.candidates
                self.preview_result = self.preview_image = None
                self.draft.component_seeds = []
                self.draft.manually_edited = False
                self.draft.segmentation_stale = False
                self._replace_working_masks(np.zeros_like(domain, np.uint8), domain)
                self.window.show_overview()
                self.stage.setCurrentIndex(2)
                self._tool("weld_select")
                self.status.setText("Select weld components, then review the fusion boundary")
                self.window._set_dirty()
            self.render()

        self.status.setText(
            "Generating provisional preview" if preview else "Generating native candidates"
        )
        self._start_job(work, done)

    def _replace_working_masks(self, envelope, domain):
        d, m = self.draft, self.window.manifest
        old = {d.envelope_layer_id, d.domain_layer_id}
        m.layers = [layer for layer in m.layers if layer.id not in old]
        for layer_id in old:
            self.window.layer_masks.pop(layer_id, None)
        for field, kind, values in (
            ("envelope_layer_id", "weld_envelope", envelope),
            ("domain_layer_id", "weld_domain", domain),
        ):
            layer = SegmentationLayer(
                name=f"{d.name} working {kind}",
                kind=kind,
                visible=False,
                scope_roi_ids=list(d.scope_roi_ids),
            )
            setattr(d, field, layer.id)
            m.layers.append(layer)
            self.window.layer_masks[layer.id] = np.asarray(values, np.uint8).copy()
        self.window._refresh_layers()

    def brush_stroke(self, tool, points):
        if tool == "weld_blur_brush" and self.active and not self.history_run_id and points:
            stroke = WeldBlurStroke(
                points=[Point(x=p.x(), y=p.y()) for p in points],
                radius_px=self.blur_radius.value(), sigma_px=self.brush_sigma.value(),
            )
            self.window.undo_stack.push(
                BlurStrokeEdit(self, [*self.draft.recipe.blur_strokes, stroke])
            )
            return
        if tool not in {"weld_brush", "weld_erase"}:
            return
        self.edit_envelope(points, erase=tool == "weld_erase")

    def polygon_finished(self, tool, points):
        if tool in {"weld_polygon_add", "weld_polygon_remove"}:
            self.edit_envelope(points, erase=tool == "weld_polygon_remove", polygon=True)

    def edit_envelope(self, points, *, erase=False, polygon=False):
        if self.history_run_id:
            self.status.setText("Return to current draft before editing")
            return
        if self.preview_result is not None:
            self.status.setText("Return to the native envelope before editing")
            return
        if self.envelope is None or self.draft.segmentation_stale:
            self.status.setText("Generate a current full-resolution envelope before editing")
            return
        if self.draft.envelope_layer_id is None:
            self._replace_working_masks(self.envelope, self.domain)
        xy = np.rint([(p.x(), p.y()) for p in points]).astype(np.int32)
        radius = self.radius.value()
        h, w = self.envelope.shape
        x0, y0 = np.maximum(xy.min(axis=0) - radius - 2, 0)
        x1, y1 = np.minimum(xy.max(axis=0) + radius + 3, [w, h])
        if x1 <= x0 or y1 <= y0:
            return
        after = self.envelope[y0:y1, x0:x1].copy()
        xy -= [x0, y0]
        value = 0 if erase else 1
        if polygon:
            cv2.fillPoly(after, [xy], value)
        elif len(xy) == 1:
            cv2.circle(after, tuple(xy[0]), radius, value, cv2.FILLED)
        else:
            cv2.polylines(after, [xy], False, value, radius * 2 + 1)
        after &= self.domain[y0:y1, x0:x1]
        self.window.undo_stack.push(
            EnvelopeEdit(
                self,
                (x0, y0, x1, y1),
                after,
                "Erase weld envelope" if erase else "Add weld envelope",
            )
        )

    def calculate(self):
        if self.history_run_id:
            return
        if not self.draft or self.draft.reference is None or self.envelope is None:
            self.status.setText("Define the surface and generate/select an envelope first")
            return
        if self.draft.segmentation_stale:
            self.status.setText("Regenerate the envelope after segmentation or domain changes")
            return
        if not self.envelope.any():
            self.status.setText("Select at least one weld component or draw its envelope")
            return
        draft = self.draft.model_copy(deep=True)
        envelope, domain = self.envelope.copy(), self.domain.copy()
        calibration = self.window.manifest.calibration
        calibration = calibration.model_copy(deep=True) if calibration else None

        def work(*, progress, cancelled):
            filled = ndi.binary_fill_holes(envelope) & domain.astype(bool)
            summary, lines = measure_weld_dilution(
                filled,
                domain,
                draft.reference,
                calibration,
                draft.sampling,
                progress=progress,
                cancelled=cancelled,
            )
            return summary, lines, filled

        def done(values):
            summary, lines, filled = values
            if self.draft.envelope_layer_id:
                self.window.layer_masks[self.draft.envelope_layer_id] = filled.astype(np.uint8)
            append_dilution_run(
                self.window.manifest,
                self.window.layer_masks,
                self.draft,
                filled,
                domain,
                summary,
                lines,
            )
            self.status.setText(
                "Calculated — inspect the boundary, profile and flags before confirming"
            )
            self.window._set_dirty()
            self.window._refresh_layers()
            self.refresh_results()
            if self.preview_result is not None:
                self.preview_result = self.preview_image = None
                self.window.show_overview()
            self.stage.setCurrentIndex(0)
            self.render()

        self._start_job(work, done)

    def confirm(self):
        run = self.run
        if not run or run.outdated or self.draft.segmentation_stale or self.history_run_id:
            self.status.setText("Calculate a current result before confirming")
            return
        run.review_status, run.reviewed_at = "confirmed", datetime.now(UTC)
        for layer in self.window.manifest.layers:
            if layer.id in run.layer_ids:
                layer.review_status = "confirmed"
        self.window.manifest.edits.append(
            EditEvent(action="confirm_weld_dilution", target_id=run.id)
        )
        self.window._set_dirty()
        self.refresh_results()

    def export(self):
        if not self.run:
            self.status.setText("Calculate a result before exporting")
            return
        path = QFileDialog.getExistingDirectory(self.window, "Export weld dilution")
        if path:
            try:
                export_dilution(
                    path, self.window.manifest, self.window.layer_masks, runs=[self.run]
                )
                self.status.setText(f"Exported dilution/{self.run.id}")
            except (OSError, ValueError, KeyError) as error:
                self.status.setText(str(error))

    def refresh_results(self):
        run = self.run
        self.confirm_button.setEnabled(bool(run and not run.outdated and not self.history_run_id))
        key = (
            (
                run.id,
                run.outdated,
                run.review_status,
                self.display_unit.currentText(),
                self.bins.value(),
                self.history_run_id,
            )
            if run
            else None
        )
        if key == self._result_cache:
            return
        self._result_cache = key
        if not run:
            self.summary_label.setText("No dilution result")
            self.tie_model.update([], 1, "px")
            self.statistics.setRowCount(0)
            for figure, canvas in zip(self.figures, self.plots):
                figure.clear()
                canvas.draw_idle()
            return
        s = run.summary
        area_unit, factor = (
            ("mm²", run.calibration.mm_per_pixel**2) if run.calibration else ("px²", 1)
        )
        status = "OUTDATED — recalculate" if run.outdated else run.review_status.upper()
        if self.history_run_id:
            status = f"Saved snapshot {run.id[:8]} — {status}"
        message = (
            f"{status}\nCross-sectional dilution: {s.dilution_percent:.4f}%\n"
            f"Penetration: {s.penetration_area_px2 * factor:.6g} {area_unit}; "
            f"reinforcement: {s.reinforcement_area_px2 * factor:.6g} {area_unit}; "
            f"total: {s.envelope_area_px2 * factor:.6g} {area_unit}\n"
            f"Samples: {s.sample_counts}. Flags: {', '.join(s.quality_flags) or 'none'}"
        )
        if s.envelope_volume_mm3 is not None:
            message += (
                f"\nEstimated volume: penetration {s.penetration_volume_mm3:.6g}, "
                f"reinforcement {s.reinforcement_volume_mm3:.6g}, "
                f"total {s.envelope_volume_mm3:.6g} mm³. {s.volume_assumption}."
            )
        self.summary_label.setText(message)
        unit = self.display_unit.currentText() if run.calibration else "px"
        factor = (
            run.calibration.mm_per_pixel * (1000 if unit == "µm" else 1)
            if run.calibration and unit != "px"
            else 1
        )
        self.tie_model.update(run.tie_lines, factor, unit)
        fields = ["count", "mean", "median", "std", "minimum", "maximum", "p5", "p95"]
        self.statistics.setColumnCount(3)
        self.statistics.setHorizontalHeaderLabels(
            ["Statistic", f"Depth ({unit})", f"Height ({unit})"]
        )
        self.statistics.setRowCount(len(fields))
        for row, field in enumerate(fields):
            self.statistics.setItem(row, 0, QTableWidgetItem(field))
            for col, stats in enumerate((s.depth_px, s.height_px), 1):
                value = getattr(stats, field)
                text = "" if value is None else f"{value * (1 if field == 'count' else factor):.6g}"
                self.statistics.setItem(row, col, QTableWidgetItem(text))
        for index, (figure, canvas) in enumerate(zip(self.figures, self.plots)):
            plot_dilution(figure, run, histogram=bool(index), unit=unit, bins=self.bins.value())
            canvas.draw_idle()

    def highlight_row(self):
        self.selected_line = self.table.currentIndex().row()
        self.render_annotations()

    def plot_picked(self, event):
        if len(event.ind):
            self.table.selectRow(int(event.ind[0]))
            self.highlight_row()

    def copy_table(self):
        if self.run:
            QApplication.clipboard().setText(tie_line_table(self.run).to_csv(sep="\t", index=False))

    def copy_summary(self):
        if self.run:
            QApplication.clipboard().setText(
                self.summary_label.text() + "\n" + self.run.summary.model_dump_json(indent=2)
            )

    def render(self, *_args):
        if not self.active or self.window.canvas._base_item is None:
            return
        canvas = self.window.canvas
        stage = 0 if self.history_run_id else self.stage.currentIndex()
        canvas.set_base_preview(self.blurred_preview if stage == 1 else None)
        if stage == 1:
            canvas.set_mask_overlay(None)
            if self.blurred_preview is None:
                self.status.setText("Use Preview blur to inspect the current Gaussian smoothing")
        elif self.preview_result is not None and stage == 2:
            canvas.set_mask_overlay(
                self.preview_result.candidates > 0, opacity=self.opacity.value()
            )
        elif stage == 2 and self.candidates is not None:
            layers = [(self.candidates, (190, 190, 190), 0.2)]
            if self.envelope is not None:
                layers.append((self.envelope, (239, 140, 40), self.opacity.value()))
            canvas.set_composite_overlays(layers)
        elif self.envelope is not None and self.overlay.isChecked():
            if self.display_reference:
                # Crop before coloring: never allocate full-resolution RGB overlays.
                mask = canvas._values_in_display_rect(self.envelope)
                width, height = canvas.preview_size
                small = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                ref, rect = self.display_reference, canvas.display_rect
                xs = np.arange(width) * rect.width() / width + rect.left() - ref.origin.x
                ys = np.arange(height) * rect.height() / height + rect.top() - ref.origin.y
                labels = np.where(ref.normal.x * xs + ref.normal.y * ys[:, None] >= 0, 1, 2)
                labels = (labels * (small > 0)).astype(np.uint8)
                canvas.set_label_overlay(
                    labels, {1: (239, 140, 40), 2: (0, 188, 212)}, self.opacity.value()
                )
            else:
                canvas.set_mask_overlay(self.envelope, opacity=self.opacity.value())
        else:
            canvas.set_mask_overlay(None)
        if stage == 1:
            self.clear_annotations()
        else:
            self.render_annotations()

    def clear_annotations(self):
        for item in self.annotation_items:
            try:
                if item.scene():
                    item.scene().removeItem(item)
            except RuntimeError:
                pass  # set_image clears the scene and deletes its Qt items.
        self.annotation_items = []

    def render_annotations(self):
        self.clear_annotations()
        if self.stage.currentIndex() == 1 and not self.history_run_id:
            return
        if not self.active or not self.overlay.isChecked():
            return
        canvas, ref = self.window.canvas, self.display_reference
        shape = (self.window.manifest.image_height, self.window.manifest.image_width)

        def add(path, color, dash=False, width=1):
            pen = QPen(QColor(color), width)
            pen.setCosmetic(True)
            if dash:
                pen.setStyle(Qt.PenStyle.DashLine)
            item = canvas.scene().addPath(path, pen)
            item.setZValue(30)
            self.annotation_items.append(item)

        def line(path, a, b):
            path.moveTo(QPointF(*a))
            path.lineTo(QPointF(*b))

        rect = canvas.display_rect
        width, height = canvas.preview_size
        for mask, color in ((self.domain, "#9ccc65"), (self.envelope, "#ffffff")):
            if mask is None:
                continue
            small = cv2.resize(
                canvas._values_in_display_rect(mask),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
            contours, _ = cv2.findContours(small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            path = QPainterPath()
            for contour in contours:
                points = contour[:, 0].astype(float)
                points = points * [rect.width() / width, rect.height() / height]
                points += [rect.left(), rect.top()]
                if len(points) < 2:
                    continue
                path.moveTo(QPointF(*points[0]))
                for point in points[1:]:
                    path.lineTo(QPointF(*point))
                path.closeSubpath()
            add(path, color)
        if ref is None:
            return
        o = np.array([ref.origin.x, ref.origin.y])
        t = np.array([ref.tangent.x, ref.tangent.y])
        n = np.array([ref.normal.x, ref.normal.y])
        path = QPainterPath()
        limits = line_bounds(o, t, shape)
        if limits:
            line(path, o + limits[0] * t, o + limits[1] * t)
        add(path, "#ffffff", dash=True, width=2)
        scale = max(0.0001, canvas.transform().m11())
        size = 25 / scale
        path = QPainterPath()
        tip = o + n * size
        line(path, o, tip)
        line(path, tip, tip - n * size * 0.3 + t * size * 0.2)
        line(path, tip, tip - n * size * 0.3 - t * size * 0.2)
        for p in ref.points:
            path.addEllipse(QPointF(p.x, p.y), 4 / scale, 4 / scale)
        add(path, "#ef8c28", width=2)
        run = self.run
        if run and (not run.outdated or self.history_run_id):
            valid = [p for p in run.tie_lines if p.status == "valid"]
            visible = canvas.mapToScene(canvas.viewport().rect()).boundingRect()
            valid = [p for p in valid if visible.contains(QPointF(p.x, p.y))]
            max_lines = max(1, canvas.viewport().width() // 20)
            path = QPainterPath()
            for p in valid[:: max(1, len(valid) // max_lines)]:
                q = np.array([p.x, p.y])
                line(path, q - p.height_px * n, q + p.depth_px * n)
            add(path, "#cfd8dc")
            if self.selected_line is not None and 0 <= self.selected_line < len(run.tie_lines):
                p = run.tie_lines[self.selected_line]
                if p.status == "valid":
                    path = QPainterPath()
                    q = np.array([p.x, p.y])
                    line(path, q - p.height_px * n, q + p.depth_px * n)
                    add(path, "#ff00ff", width=3)
