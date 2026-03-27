from __future__ import annotations
import csv
from typing import List, Optional, Tuple, Union
from PyQt5.QtWidgets import (QWidget, QSizePolicy, QFileDialog, QToolButton,
                              QHBoxLayout, QVBoxLayout, QStyle, QMenu, QAction)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPen, QPixmap

from .canvas import _PlotCanvas
from .items import _LineItem, _ScatterItem, _FitItem, _InfLine
from .sidebar import SidebarLabel
from .i18n import tr
from .math_utils import get_fit_modes, get_fit_mode
from .palette import next_line_color, next_scatter_color

_AnyItem = Union[_LineItem, _ScatterItem]


class ChartWidget(QWidget):
    def __init__(self, parent=None, *,
                 show_toolbar: bool = True,
                 show_legend: bool = False,
                 show_sidebar: bool = False,
                 font: Optional[QFont] = None):
        super().__init__(parent)
        self._lines: List[_LineItem] = []
        self._scatters: List[_ScatterItem] = []
        self._fits: List[_FitItem] = []
        self._inflines: List[_InfLine] = []
        self._label_left = ""
        self._label_bottom = ""
        self._font = font or QFont("Arial", 8)
        self._vx0 = 0.0; self._vx1 = 1.0
        self._vy0 = 0.0; self._vy1 = 1.0
        self._show_legend = show_legend
        self._active_fit_key: Optional[str] = None
        self._canvas = _PlotCanvas(self)
        self._toolbar_layout = self._build_toolbar()
        self._toolbar_widget = QWidget(self)
        self._toolbar_widget.setLayout(self._toolbar_layout)
        self._sidebar = SidebarLabel(self) if show_sidebar else None
        self._toolbar_widget.setVisible(show_toolbar)
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        if self._sidebar:
            content.addWidget(self._sidebar)
        content.addWidget(self._canvas, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._toolbar_widget)
        root.addLayout(content, 1)
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _build_toolbar(self) -> QHBoxLayout:
        self._btn_fit = self._make_btn("SP_FileDialogContentsView",
                                       "chart_widget.btn_fit",
                                       "chart_widget.btn_fit_tip",
                                       self.autofit)
        self._btn_stats = self._make_btn("SP_FileDialogInfoView",
                                         "chart_widget.btn_analytics",
                                         "chart_widget.btn_analytics_tip",
                                         self._canvas.toggleAnalytics)
        self._btn_csv = self._make_btn("SP_DialogSaveButton",
                                       "chart_widget.btn_csv",
                                       "chart_widget.btn_csv_tip",
                                       self.exportCsv)
        self._btn_img = self._make_btn("SP_DialogSaveButton",
                                       "chart_widget.btn_img",
                                       "chart_widget.btn_img_tip",
                                       self.exportImage)
        self._btn_fit_mode = QToolButton(self)
        self._btn_fit_mode.setToolTip(tr("chart_widget.btn_fit_mode_tip"))
        self._btn_fit_mode.setFixedHeight(24)
        self._btn_fit_mode.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._btn_fit_mode.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._fit_menu = QMenu(self._btn_fit_mode)
        for mode in get_fit_modes():
            act = QAction(mode.label, self)
            act.setData(mode.key)
            act.triggered.connect(lambda checked, k=mode.key: self._on_fit_mode_selected(k))
            self._fit_menu.addAction(act)
        self._btn_fit_mode.setMenu(self._fit_menu)
        self._update_fit_mode_label()
        tb = QHBoxLayout()
        tb.setContentsMargins(2, 2, 2, 2)
        tb.setSpacing(2)
        tb.addStretch()
        for w in (self._btn_fit, self._btn_stats, self._btn_fit_mode, self._btn_csv, self._btn_img):
            tb.addWidget(w)
        return tb

    def _make_btn(self, icon_sp: str, lbl_key: str, tip_key: str, slot) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(tr(lbl_key))
        btn.setToolTip(tr(tip_key))
        btn.setIcon(self.style().standardIcon(getattr(QStyle.StandardPixmap, icon_sp)))
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setFixedHeight(24)
        btn.clicked.connect(slot)
        return btn

    def _on_fit_mode_selected(self, key: str):
        self._active_fit_key = key
        self._update_fit_mode_label()
        for fit in self._fits:
            fit.setModeKey(key)
        self._canvas.update()

    def _update_fit_mode_label(self):
        if self._active_fit_key is None:
            self._btn_fit_mode.setText("Approx")
            return
        mode = get_fit_mode(self._active_fit_key)
        lbl = mode.label if mode else self._active_fit_key
        self._btn_fit_mode.setText(f"{lbl} ▾")

    def setToolbarVisible(self, visible: bool):
        self._toolbar_widget.setVisible(visible)

    def setSidebarVisible(self, visible: bool):
        if self._sidebar:
            self._sidebar.setVisible(visible)

    def sidebar(self) -> Optional[SidebarLabel]:
        return self._sidebar

    def setLabel(self, side: str, text: str):
        if side == "left": self._label_left = text
        elif side == "bottom": self._label_bottom = text
        self._canvas.update()

    def setFont(self, font: QFont):
        self._font = font
        self._canvas.update()

    def setLegendVisible(self, visible: bool):
        self._show_legend = visible
        self._canvas.update()

    def plot(self, color: Optional[str] = None, width: int = 2,
             label: str = "", dashed: bool = False) -> _LineItem:
        c = color or next_line_color()
        pen = QPen(QColor(c), width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        item = _LineItem(self, pen, label)
        self._lines.append(item)
        return item

    def addScatter(self, size: int = 10, color: Optional[str] = None,
                   label: str = "") -> _ScatterItem:
        c = color or next_scatter_color()
        item = _ScatterItem(self, size, QColor(c), label)
        self._scatters.append(item)
        return item

    def addFit(self, source: _AnyItem, mode_key: Optional[str] = None,
               color: Optional[str] = None, width: int = 2,
               dashed: bool = True, label: str = "") -> _FitItem:
        key = mode_key or self._active_fit_key or "linear"
        c = color or next_line_color()
        pen = QPen(QColor(c), width,
                   Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        fit = _FitItem(self, source, key, pen, label)
        self._fits.append(fit)
        if self._active_fit_key is None:
            self._active_fit_key = key
            self._update_fit_mode_label()
        self._canvas.update()
        return fit

    def addLine(self, y: Optional[float] = None, x: Optional[float] = None,
                color: str = "#f39c12", width: int = 1,
                dashed: bool = True) -> _InfLine:
        horiz = y is not None
        val = y if horiz else (x if x is not None else 0.0)
        pen = QPen(QColor(color), width,
                   Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        ln = _InfLine(self, horiz, val, pen)
        self._inflines.append(ln)
        return ln

    def removeItem(self, item):
        for lst in (self._lines, self._scatters, self._fits, self._inflines):
            if item in lst:
                lst.remove(item)
        self._canvas.update()

    def clearAll(self):
        self._lines.clear()
        self._scatters.clear()
        self._fits.clear()
        self._inflines.clear()
        self._canvas.update()

    def _all_xy(self) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for item in self._lines + self._scatters:
            xs.extend(item.xs); ys.extend(item.ys)
        for ln in self._inflines:
            if ln.visible:
                (ys if ln.horizontal else xs).append(ln.value)
        return xs, ys

    def _data_bounds(self) -> Tuple[float, float, float, float]:
        xs, ys = self._all_xy()
        if not xs or not ys:
            return 0.0, 1.0, 0.0, 1.0
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if x0 == x1: x0 -= 1.0; x1 += 1.0
        if y0 == y1: y0 -= 1.0; y1 += 1.0
        px = (x1 - x0) * 0.05
        py = (y1 - y0) * 0.08
        return x0 - px, x1 + px, y0 - py, y1 + py

    def _schedule_autofit(self):
        self.autofit()

    def autofit(self):
        x0, x1, y0, y1 = self._data_bounds()
        self._vx0 = x0; self._vx1 = x1
        self._vy0 = y0; self._vy1 = y1
        self._canvas.update()

    def update(self):
        self._canvas.update()
        super().update()

    def refreshFitMenu(self):
        self._fit_menu.clear()
        for mode in get_fit_modes():
            act = QAction(mode.label, self)
            act.setData(mode.key)
            act.triggered.connect(lambda checked, k=mode.key: self._on_fit_mode_selected(k))
            self._fit_menu.addAction(act)

    def exportCsv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, tr("chart_widget.csv_title"), "chart_data.csv",
            tr("chart_widget.csv_filter"))
        if not path:
            return
        series = []
        for i, it in enumerate(self._lines):
            if it.xs:
                n = it.label or f"line{i}"
                series.append((f"{n}_x", f"{n}_y", it.xs, it.ys))
        for i, it in enumerate(self._scatters):
            if it.xs:
                n = it.label or f"scatter{i}"
                series.append((f"{n}_x", f"{n}_y", it.xs, it.ys))
        if not series:
            return
        max_rows = max(len(s[2]) for s in series)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([col for s in series for col in (s[0], s[1])])
            for row in range(max_rows):
                w.writerow([
                    v for s in series
                    for v in (s[2][row] if row < len(s[2]) else "",
                              s[3][row] if row < len(s[3]) else "")
                ])

    def exportImage(self):
        path, _ = QFileDialog.getSaveFileName(
            self, tr("chart_widget.img_title"), "chart.png",
            tr("chart_widget.img_filter"))
        if not path:
            return
        self._canvas.grab_image().save(path)

    def grabImage(self) -> QPixmap:
        return self._canvas.grab_image()