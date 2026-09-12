"""Interactive saved particle grouping and filtering controls."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gst_image.analysis.groups import generate_particle_groups, validate_particle_groups
from gst_image.models import (
    ParticleCriteria,
    ParticleGroup,
    ParticleGrouping,
    ParticleRecord,
    ParticleSizeMetric,
)
from gst_image_app.range_slider import MetricRangeControl


class ParticleGroupingPanel(QWidget):
    draftChanged = Signal(object)
    saveRequested = Signal(object)
    deleteRequested = Signal(str)
    plotRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self._source_layer_id: str | None = None
        self._particles: list[ParticleRecord] = []
        self._calibrated = False
        self._saved: dict[str, ParticleGrouping] = {}
        self._draft: ParticleGrouping | None = None
        self._size_domain = (0.0, 1.0)
        self._build_ui()
        self.setEnabled(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        saved_row = QHBoxLayout()
        self.scheme_combo = QComboBox()
        self.scheme_combo.currentIndexChanged.connect(self._scheme_changed)
        saved_row.addWidget(QLabel("Saved scheme"))
        saved_row.addWidget(self.scheme_combo, 1)
        layout.addLayout(saved_row)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Grouping name")
        self.name_edit.textEdited.connect(self._emit_draft)
        layout.addWidget(self.name_edit)
        scheme_buttons = QHBoxLayout()
        for text, slot in (
            ("New", self.new_grouping),
            ("Save", self._save),
            ("Revert", self._revert),
            ("Delete", self._delete),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            scheme_buttons.addWidget(button)
        layout.addLayout(scheme_buttons)

        metric_form = QFormLayout()
        self.size_metric = QComboBox()
        self.size_metric.addItem("Equivalent radius", ParticleSizeMetric.EQUIVALENT_RADIUS)
        self.size_metric.addItem("Equivalent diameter", ParticleSizeMetric.EQUIVALENT_DIAMETER)
        self.size_metric.addItem("Area", ParticleSizeMetric.AREA)
        self.size_metric.currentIndexChanged.connect(self._metric_changed)
        metric_form.addRow("Size measurement", self.size_metric)
        self.size_unit = QComboBox()
        self.size_unit.addItem("Millimetres", "mm")
        self.size_unit.addItem("Pixels", "px")
        self.size_unit.currentIndexChanged.connect(self._metric_changed)
        metric_form.addRow("Size unit", self.size_unit)
        layout.addLayout(metric_form)

        filter_box = QGroupBox("Analysis filter (non-destructive)")
        filter_box.setCheckable(True)
        filter_box.setChecked(False)
        filter_box.toggled.connect(self._filter_changed)
        self.filter_box = filter_box
        filter_layout = QVBoxLayout(filter_box)
        self.filter_size = QCheckBox("Filter by size")
        self.filter_size.setChecked(True)
        self.filter_size.toggled.connect(self._filter_changed)
        self.filter_circularity = QCheckBox("Filter by circularity")
        self.filter_circularity.toggled.connect(self._filter_changed)
        filter_layout.addWidget(self.filter_size)
        self.filter_size_range = MetricRangeControl()
        self.filter_size_range.rangeChanged.connect(self._filter_changed)
        filter_layout.addWidget(self.filter_size_range)
        filter_layout.addWidget(self.filter_circularity)
        self.filter_circularity_range = MetricRangeControl()
        self.filter_circularity_range.set_domain(0, 1)
        self.filter_circularity_range.rangeChanged.connect(self._filter_changed)
        filter_layout.addWidget(self.filter_circularity_range)
        self.show_filtered = QCheckBox("Show filtered particles in grey")
        self.show_filtered.toggled.connect(self._filter_changed)
        self.show_unclassified = QCheckBox("Show unclassified particles")
        self.show_unclassified.setChecked(True)
        self.show_unclassified.toggled.connect(self._filter_changed)
        filter_layout.addWidget(self.show_filtered)
        filter_layout.addWidget(self.show_unclassified)
        layout.addWidget(filter_box)

        split_box = QGroupBox("Generate colored groups")
        split_form = QFormLayout(split_box)
        self.size_bins = QSpinBox()
        self.size_bins.setRange(1, 12)
        self.size_bins.setValue(3)
        self.circularity_bins = QSpinBox()
        self.circularity_bins.setRange(1, 12)
        self.circularity_bins.setValue(1)
        split_form.addRow("Size groups", self.size_bins)
        split_form.addRow("Circularity groups", self.circularity_bins)
        generate = QPushButton("Generate / replace groups")
        generate.clicked.connect(self._generate_groups)
        split_form.addRow(generate)
        layout.addWidget(split_box)

        self.groups_table = QTableWidget(0, 7)
        self.groups_table.setHorizontalHeaderLabels(
            ["On", "Color", "Name", "Min size", "Max size", "Min circ.", "Max circ."]
        )
        self.groups_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.groups_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.groups_table.itemSelectionChanged.connect(self._selected_group_changed)
        self.groups_table.itemChanged.connect(self._table_changed)
        self.groups_table.cellDoubleClicked.connect(self._cell_double_clicked)
        layout.addWidget(self.groups_table)
        group_buttons = QHBoxLayout()
        add = QPushButton("Add group")
        add.clicked.connect(self._add_group)
        remove = QPushButton("Remove group")
        remove.clicked.connect(self._remove_group)
        color = QPushButton("Set color")
        color.clicked.connect(self._set_color)
        group_buttons.addWidget(add)
        group_buttons.addWidget(remove)
        group_buttons.addWidget(color)
        layout.addLayout(group_buttons)

        selected_box = QGroupBox("Selected group range")
        selected_layout = QVBoxLayout(selected_box)
        selected_layout.addWidget(QLabel("Size minimum / maximum"))
        self.group_size_range = MetricRangeControl()
        self.group_size_range.rangeChanged.connect(self._selected_range_changed)
        selected_layout.addWidget(self.group_size_range)
        selected_layout.addWidget(QLabel("Circularity minimum / maximum"))
        self.group_circularity_range = MetricRangeControl()
        self.group_circularity_range.set_domain(0, 1)
        self.group_circularity_range.rangeChanged.connect(self._selected_range_changed)
        selected_layout.addWidget(self.group_circularity_range)
        layout.addWidget(selected_box)

        self.preview_status = QLabel("Run particle analysis to create groups")
        self.preview_status.setWordWrap(True)
        layout.addWidget(self.preview_status)
        plot = QPushButton("Open color plots")
        plot.clicked.connect(self.plotRequested)
        layout.addWidget(plot)

        self.statistics_table = QTableWidget(0, 8)
        self.statistics_table.setHorizontalHeaderLabels(
            [
                "Group",
                "Count",
                "Count %",
                "Area px²",
                "Area fraction %",
                "Mean size",
                "Median size",
                "Mean circularity",
            ]
        )
        self.statistics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.statistics_table)

    def set_context(
        self,
        source_layer_id: str | None,
        particles: list[ParticleRecord],
        *,
        calibrated: bool,
        groupings: list[ParticleGrouping],
        active_id: str | None,
    ) -> None:
        self._loading = True
        self._source_layer_id = source_layer_id
        self._particles = particles
        self._calibrated = calibrated
        self._saved = {item.id: item.model_copy(deep=True) for item in groupings}
        self.scheme_combo.clear()
        for grouping in groupings:
            self.scheme_combo.addItem(grouping.name, grouping.id)
        index = self.scheme_combo.findData(active_id)
        if index < 0 and self.scheme_combo.count():
            index = 0
        self.scheme_combo.setCurrentIndex(index)
        self.setEnabled(bool(source_layer_id and particles))
        chosen = self._saved.get(self.scheme_combo.currentData())
        unit = self._grouping_unit(chosen) if chosen else ("mm" if calibrated else "px")
        self.size_unit.clear()
        if calibrated or unit == "mm":
            self.size_unit.addItem("Millimetres", "mm")
        self.size_unit.addItem("Pixels", "px")
        self.size_unit.setCurrentIndex(max(0, self.size_unit.findData(unit)))
        self._draft = chosen.model_copy(deep=True) if chosen else self._default_grouping()
        metric = self._grouping_metric(self._draft)
        self.size_metric.setCurrentIndex(self.size_metric.findData(metric))
        self._update_domains(metric, self.size_unit.currentData())
        self._load_draft()
        self.statistics_table.setRowCount(0)
        self.preview_status.setText(
            "Adjust a range for a quick preview, then save the grouping scheme."
            if source_layer_id and particles
            else "Select a particle result layer to create groups."
        )
        self._loading = False

    def current_grouping(self) -> ParticleGrouping | None:
        self._sync_draft()
        return self._draft.model_copy(deep=True) if self._draft else None

    def set_statistics(self, values: dict[str, dict]) -> None:
        self.statistics_table.horizontalHeaderItem(3).setText(
            "Area mm²" if self._calibrated else "Area px²"
        )
        self.statistics_table.setRowCount(len(values))
        for row, (name, stats) in enumerate(values.items()):
            size = stats.get("size", {})
            area_fraction = stats.get("area_fraction")
            cells = [
                name,
                str(stats.get("count", 0)),
                f"{stats.get('count_percent', 0):.3g}",
                self._format_optional(
                    stats.get("area_mm2") if self._calibrated else stats.get("area_px")
                ),
                "" if area_fraction is None else f"{100 * area_fraction:.6g}",
                self._format_optional(size.get("mean")),
                self._format_optional(size.get("median")),
                self._format_optional(stats.get("circularity", {}).get("mean")),
            ]
            for column, value in enumerate(cells):
                self.statistics_table.setItem(row, column, QTableWidgetItem(value))

    def set_preview_counts(self, included: int, total: int, unclassified: int) -> None:
        self.preview_status.setText(
            f"Quick preview: {included} of {total} particles included; "
            f"{unclassified} included particles unclassified. Save to keep this scheme."
        )

    @staticmethod
    def _format_optional(value) -> str:
        return "" if value is None else f"{value:.6g}"

    def _default_grouping(self) -> ParticleGrouping:
        metric = self.size_metric.currentData() or ParticleSizeMetric.EQUIVALENT_RADIUS
        unit = self.size_unit.currentData() or ("mm" if self._calibrated else "px")
        self._update_domains(metric, unit)
        size_min, size_max = self._size_domain
        group = ParticleGroup(
            name="Included particles",
            color="#42a5f5",
            size_metric=metric,
            size_unit=unit,
            size_min=size_min,
            size_max=size_max,
            circularity_min=0,
            circularity_max=1,
        )
        return ParticleGrouping(source_layer_id=self._source_layer_id, groups=[group])

    @staticmethod
    def _grouping_metric(grouping: ParticleGrouping) -> ParticleSizeMetric:
        if grouping.groups:
            return grouping.groups[0].size_metric
        if grouping.filter_criteria:
            return grouping.filter_criteria.size_metric
        return ParticleSizeMetric.EQUIVALENT_RADIUS

    @staticmethod
    def _grouping_unit(grouping: ParticleGrouping) -> str:
        if grouping.groups:
            return grouping.groups[0].size_unit
        if grouping.filter_criteria:
            return grouping.filter_criteria.size_unit
        return "px"

    def _update_domains(self, metric: ParticleSizeMetric, unit: str | None = None) -> None:
        unit = unit or self.size_unit.currentData() or (
            "mm" if self._calibrated else "px"
        )
        criteria = ParticleCriteria(size_metric=metric, size_unit=unit)
        values = [
            value
            for particle in self._particles
            if (value := criteria.size_value(particle)) is not None
        ]
        self._size_domain = (min(values), max(values)) if values else (0.0, 1.0)
        if self._size_domain[1] <= self._size_domain[0]:
            center = self._size_domain[0]
            pad = max(1e-6, abs(center) * 0.01)
            self._size_domain = (max(0.0, center - pad), center + pad)
        suffix = self._unit_suffix(metric, unit)
        self.filter_size_range.set_domain(*self._size_domain, suffix=suffix)
        self.group_size_range.set_domain(*self._size_domain, suffix=suffix)

    @staticmethod
    def _unit_suffix(metric: ParticleSizeMetric, unit: str) -> str:
        if metric == ParticleSizeMetric.AREA:
            return f" {unit}²"
        return f" {unit}"

    def _load_draft(self) -> None:
        if self._draft is None:
            return
        self._loading = True
        self.name_edit.setText(self._draft.name)
        filter_criteria = self._draft.filter_criteria
        self.filter_box.setChecked(filter_criteria is not None)
        self.filter_size.setChecked(
            bool(filter_criteria and (filter_criteria.size_min is not None or filter_criteria.size_max is not None))
        )
        self.filter_circularity.setChecked(
            bool(
                filter_criteria
                and (
                    filter_criteria.circularity_min is not None
                    or filter_criteria.circularity_max is not None
                )
            )
        )
        self.filter_size_range.set_values(
            filter_criteria.size_min if filter_criteria and filter_criteria.size_min is not None else self._size_domain[0],
            filter_criteria.size_max if filter_criteria and filter_criteria.size_max is not None else self._size_domain[1],
        )
        self.filter_circularity_range.set_values(
            filter_criteria.circularity_min if filter_criteria and filter_criteria.circularity_min is not None else 0,
            filter_criteria.circularity_max if filter_criteria and filter_criteria.circularity_max is not None else 1,
        )
        self.show_filtered.setChecked(self._draft.show_filtered)
        self.show_unclassified.setChecked(self._draft.show_unclassified)
        self._refresh_groups()
        self._loading = False

    def _refresh_groups(self) -> None:
        self.groups_table.blockSignals(True)
        self.groups_table.setRowCount(0)
        if self._draft:
            for group in self._draft.groups:
                row = self.groups_table.rowCount()
                self.groups_table.insertRow(row)
                enabled = QTableWidgetItem()
                enabled.setCheckState(
                    Qt.CheckState.Checked if group.enabled else Qt.CheckState.Unchecked
                )
                enabled.setData(Qt.ItemDataRole.UserRole, group.id)
                color = QTableWidgetItem(group.color)
                color.setBackground(QColor(group.color))
                values = [
                    enabled,
                    color,
                    QTableWidgetItem(group.name),
                    QTableWidgetItem(f"{group.size_min:.17g}" if group.size_min is not None else ""),
                    QTableWidgetItem(f"{group.size_max:.17g}" if group.size_max is not None else ""),
                    QTableWidgetItem(
                        f"{group.circularity_min:.17g}" if group.circularity_min is not None else ""
                    ),
                    QTableWidgetItem(
                        f"{group.circularity_max:.17g}" if group.circularity_max is not None else ""
                    ),
                ]
                for column, item in enumerate(values):
                    self.groups_table.setItem(row, column, item)
        self.groups_table.blockSignals(False)
        if self.groups_table.rowCount():
            self.groups_table.selectRow(0)

    def _sync_draft(self) -> None:
        if self._draft is None or self._loading:
            return
        self._draft.name = self.name_edit.text().strip() or "Particle grouping"
        self._draft.show_filtered = self.show_filtered.isChecked()
        self._draft.show_unclassified = self.show_unclassified.isChecked()
        metric = self.size_metric.currentData()
        unit = self.size_unit.currentData() or ("mm" if self._calibrated else "px")
        if self.filter_box.isChecked() and (
            self.filter_size.isChecked() or self.filter_circularity.isChecked()
        ):
            size_min, size_max = self.filter_size_range.values()
            circularity_min, circularity_max = self.filter_circularity_range.values()
            self._draft.filter_criteria = ParticleCriteria(
                size_metric=metric,
                size_unit=unit,
                size_min=size_min if self.filter_size.isChecked() else None,
                size_max=size_max if self.filter_size.isChecked() else None,
                circularity_min=(
                    circularity_min if self.filter_circularity.isChecked() else None
                ),
                circularity_max=(
                    circularity_max if self.filter_circularity.isChecked() else None
                ),
            )
        else:
            self._draft.filter_criteria = None
        groups: list[ParticleGroup] = []
        for row in range(self.groups_table.rowCount()):
            group_id = self.groups_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            groups.append(
                ParticleGroup(
                    id=group_id,
                    name=self.groups_table.item(row, 2).text().strip() or f"Group {row + 1}",
                    color=self.groups_table.item(row, 1).text(),
                    enabled=self.groups_table.item(row, 0).checkState() == Qt.CheckState.Checked,
                    size_metric=metric,
                    size_unit=unit,
                    size_min=self._float_cell(row, 3),
                    size_max=self._float_cell(row, 4),
                    circularity_min=self._float_cell(row, 5),
                    circularity_max=self._float_cell(row, 6),
                )
            )
        self._draft.groups = groups
        self._draft.updated_at = datetime.now(UTC)

    def _float_cell(self, row: int, column: int) -> float | None:
        text = self.groups_table.item(row, column).text().strip()
        return float(text) if text else None

    def _emit_draft(self, *_args) -> None:
        if self._loading:
            return
        try:
            self._sync_draft()
        except ValueError:
            return
        if self._draft:
            self.draftChanged.emit(self._draft.model_copy(deep=True))

    def _scheme_changed(self) -> None:
        if self._loading:
            return
        chosen = self._saved.get(self.scheme_combo.currentData())
        if chosen:
            self._draft = chosen.model_copy(deep=True)
            metric = self._grouping_metric(self._draft)
            unit = self._grouping_unit(self._draft)
            self.size_unit.setCurrentIndex(max(0, self.size_unit.findData(unit)))
            self.size_metric.setCurrentIndex(self.size_metric.findData(metric))
            self._update_domains(metric, self.size_unit.currentData())
            self._load_draft()
            self._emit_draft()

    def new_grouping(self) -> None:
        self._draft = self._default_grouping()
        self._draft.name = f"Particle grouping {len(self._saved) + 1}"
        self.scheme_combo.setCurrentIndex(-1)
        self._load_draft()
        self._emit_draft()

    def _save(self) -> None:
        if self._draft is None:
            return
        try:
            self._sync_draft()
            validate_particle_groups(self._draft.groups)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid particle grouping", str(error))
            return
        self.saveRequested.emit(self._draft.model_copy(deep=True))

    def _revert(self) -> None:
        saved = self._saved.get(self.scheme_combo.currentData())
        if saved:
            self._draft = saved.model_copy(deep=True)
            self._load_draft()
            self._emit_draft()
        else:
            self.new_grouping()

    def _delete(self) -> None:
        grouping_id = self.scheme_combo.currentData()
        if grouping_id:
            self.deleteRequested.emit(grouping_id)

    def _metric_changed(self) -> None:
        if self._loading or self._draft is None:
            return
        self._update_domains(self.size_metric.currentData(), self.size_unit.currentData())
        self.filter_size_range.set_values(*self._size_domain)
        self._generate_groups()

    def _filter_changed(self, *_args) -> None:
        self.filter_size_range.setEnabled(self.filter_size.isChecked())
        self.filter_circularity_range.setEnabled(self.filter_circularity.isChecked())
        self._emit_draft()

    def _generate_groups(self) -> None:
        if self._draft is None:
            return
        size_bounds = (
            self.filter_size_range.values()
            if self.filter_box.isChecked() and self.filter_size.isChecked()
            else self._size_domain
        )
        circularity_bounds = (
            self.filter_circularity_range.values()
            if self.filter_box.isChecked() and self.filter_circularity.isChecked()
            else (0.0, 1.0)
        )
        size_edges = np.linspace(*size_bounds, self.size_bins.value() + 1).tolist()
        circularity_edges = np.linspace(
            *circularity_bounds, self.circularity_bins.value() + 1
        ).tolist()
        try:
            self._draft.groups = generate_particle_groups(
                size_edges,
                circularity_edges,
                size_metric=self.size_metric.currentData(),
                size_unit=self.size_unit.currentData(),
            )
        except ValueError as error:
            QMessageBox.warning(self, "Cannot generate groups", str(error))
            return
        self._refresh_groups()
        self._emit_draft()

    def _add_group(self) -> None:
        if self._draft is None:
            return
        size_min, size_max = self._size_domain
        self._draft.groups.append(
            ParticleGroup(
                name=f"Group {len(self._draft.groups) + 1}",
                color="#42a5f5",
                size_metric=self.size_metric.currentData(),
                size_unit=self.size_unit.currentData(),
                size_min=size_min,
                size_max=size_max,
                circularity_min=0,
                circularity_max=1,
            )
        )
        self._refresh_groups()
        self.groups_table.selectRow(self.groups_table.rowCount() - 1)
        self._emit_draft()

    def _remove_group(self) -> None:
        row = self.groups_table.currentRow()
        if row < 0:
            return
        self.groups_table.removeRow(row)
        self._emit_draft()

    def _selected_group_changed(self) -> None:
        row = self.groups_table.currentRow()
        if row < 0:
            return
        try:
            self._loading = True
            self.group_size_range.set_values(
                self._float_cell(row, 3) if self._float_cell(row, 3) is not None else self._size_domain[0],
                self._float_cell(row, 4) if self._float_cell(row, 4) is not None else self._size_domain[1],
            )
            self.group_circularity_range.set_values(
                self._float_cell(row, 5) if self._float_cell(row, 5) is not None else 0,
                self._float_cell(row, 6) if self._float_cell(row, 6) is not None else 1,
            )
        except (AttributeError, ValueError):
            return
        finally:
            self._loading = False

    def _selected_range_changed(self, *_args) -> None:
        if self._loading:
            return
        row = self.groups_table.currentRow()
        if row < 0:
            return
        self.groups_table.blockSignals(True)
        for column, value in zip(
            (3, 4), self.group_size_range.values(), strict=True
        ):
            self.groups_table.item(row, column).setText(f"{value:.17g}")
        for column, value in zip(
            (5, 6), self.group_circularity_range.values(), strict=True
        ):
            self.groups_table.item(row, column).setText(f"{value:.17g}")
        self.groups_table.blockSignals(False)
        self._emit_draft()

    def _table_changed(self, _item: QTableWidgetItem) -> None:
        self._selected_group_changed()
        self._emit_draft()

    def _cell_double_clicked(self, _row: int, column: int) -> None:
        if column == 1:
            self._set_color()

    def _set_color(self) -> None:
        row = self.groups_table.currentRow()
        if row < 0:
            return
        item = self.groups_table.item(row, 1)
        color = QColorDialog.getColor(QColor(item.text()), self, "Particle group color")
        if not color.isValid():
            return
        item.setText(color.name())
        item.setBackground(color)
        self._emit_draft()
