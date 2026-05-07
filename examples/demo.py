from __future__ import annotations
import sys
import math
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
                              QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
                              QCheckBox, QSlider, QSpinBox, QPushButton,
                              QComboBox, QStatusBar, QLineEdit, QToolButton,
                              QScrollArea, QFrame, QColorDialog, QSplitter)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QPen
from pyqt5_chart_widget import (ChartWidget, FitMode, register_fit_mode,
                                set_palette, update_strings,
                                _FunctionItem, _RulerItem,
                                _DerivativeItem, _IntegralItem,
                                _HistogramItem, _SpectrumItem, _ErrorBandItem,
                                derivative, second_derivative,
                                cumulative_integral, fft_spectrum_numpy,
                                histogram, autocorrelation,
                                normalize, peak_find, moving_std)


def _damped_fit(x_pts, y_pts, x_eval):
    if not x_pts:
        return [0.0] * len(x_eval)
    amp = max(abs(y) for y in y_pts) if y_pts else 1.0
    freq, decay = 0.4, 0.05
    return [amp * math.exp(-decay * xi) * math.cos(freq * xi) for xi in x_eval]


register_fit_mode(FitMode("damped", "Damped cosine", _damped_fit, min_points=3))
update_strings({"chart_widget.btn_fit": "Fit"})

_SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_SAFE_MATH["abs"] = abs


def _compile_expression(expr: str):
    """Compile expression string to callable f(xs) -> List[float|None]; returns None on syntax error."""
    expr = expr.strip()
    if not expr:
        return None
    try:
        code = compile(expr, "<expr>", "eval")
    except SyntaxError:
        return None
    def fn(xs):
        results = []
        local_ns = dict(_SAFE_MATH)
        for x in xs:
            local_ns["x"] = x
            try:
                v = eval(code, {"__builtins__": {}}, local_ns)
                fv = float(v)
                results.append(None if not math.isfinite(fv) else fv)
            except Exception:
                results.append(None)
        return results
    return fn


_PRESET_EXPRESSIONS = [
    ("sin(x)", "#3498db"),
    ("cos(x)", "#2ecc71"),
    ("tan(x)", "#e74c3c"),
    ("x**2", "#9b59b6"),
    ("x**3 - 3*x", "#f39c12"),
    ("1/x", "#1abc9c"),
    ("sqrt(abs(x))", "#e67e22"),
    ("sin(x)/x if x != 0 else 1", "#e91e63"),
]


class _FunctionRow(QWidget):
    """Single row in the graphing calculator: expression input, colour button, visibility toggle, remove."""

    def __init__(self, chart: ChartWidget, expr: str, color: str, parent=None):
        super().__init__(parent)
        self._chart = chart
        self._chart.setOriginAxesVisible(True)
        self._color = color
        self._fn_item: _FunctionItem = None
        self._edit = QLineEdit(expr)
        self._edit.setFont(QFont("Monospace", 12))
        self._edit.setPlaceholderText("e.g. sin(x)*exp(-0.1*x)")
        self._edit.returnPressed.connect(self._apply)
        self._edit.textChanged.connect(self._on_text_changed)
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(24, 24)
        self._color_btn.setToolTip("Pick colour")
        self._color_btn.clicked.connect(self._pick_color)
        self._update_color_btn()
        self._vis_btn = QToolButton()
        self._vis_btn.setCheckable(True)
        self._vis_btn.setChecked(True)
        self._vis_btn.setText("ON")
        self._vis_btn.setFixedWidth(36)
        self._vis_btn.toggled.connect(self._on_vis_toggled)
        self._remove_btn = QToolButton()
        self._remove_btn.setText("x")
        self._remove_btn.setFixedWidth(24)
        self._remove_btn.clicked.connect(self._remove)
        self._status = QLabel()
        self._status.setFixedWidth(16)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(4)
        for w in (self._color_btn, self._edit, self._status, self._vis_btn, self._remove_btn):
            row.addWidget(w, 1 if w is self._edit else 0)
        self._fn_item = self._build_item(expr)

    def _build_item(self, expr: str) -> _FunctionItem:
        fn = _compile_expression(expr)
        pen = QPen(QColor(self._color), 6)
        item = _FunctionItem(self._chart, fn or (lambda xs: [None] * len(xs)), pen, expr or "f(x)")
        self._chart._functions.append(item)
        self._chart._canvas.update()
        self._set_status(fn is not None)
        return item

    def _apply(self):
        expr = self._edit.text().strip()
        fn = _compile_expression(expr)
        self._set_status(fn is not None)
        if fn is not None and self._fn_item is not None:
            self._fn_item.label = expr or "f(x)"
            self._fn_item.setFunction(fn)

    def _on_text_changed(self, _text):
        expr = self._edit.text().strip()
        fn = _compile_expression(expr)
        self._set_status(fn is not None or not expr)
        if fn is not None and self._fn_item is not None:
            self._fn_item.label = expr or "f(x)"
            self._fn_item.setFunction(fn)

    def _set_status(self, ok: bool):
        if not self._edit.text().strip():
            self._status.setText("")
            self._status.setToolTip("")
        elif ok:
            self._status.setText("OK")
            self._status.setStyleSheet("color: green; font-size: 9px;")
            self._status.setToolTip("Expression is valid")
        else:
            self._status.setText("ERR")
            self._status.setStyleSheet("color: red; font-size: 9px;")
            self._status.setToolTip("Syntax error or invalid expression")

    def _on_vis_toggled(self, checked: bool):
        self._vis_btn.setText("ON" if checked else "OFF")
        if self._fn_item is not None:
            self._fn_item.setVisible(checked)

    def _pick_color(self):
        col = QColorDialog.getColor(QColor(self._color), self, "Pick colour")
        if col.isValid():
            self._color = col.name()
            self._update_color_btn()
            if self._fn_item is not None:
                self._fn_item.pen.setColor(col)
                self._fn_item.invalidateCache()
                self._chart._canvas.update()

    def _update_color_btn(self):
        self._color_btn.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid #666; border-radius: 3px;")

    def _remove(self):
        if self._fn_item is not None:
            self._chart.removeItem(self._fn_item)
            self._fn_item = None
        parent = self.parent()
        self.setParent(None)
        self.deleteLater()
        if parent is not None:
            parent.layout().update()


class GraphingCalculatorTab(QWidget):
    """
    Graphing calculator: user-typed expressions, ruler, viewport readout, preset library.

    Demonstrates addFunction(), addRuler(), onViewportChanged(), setViewport().
    """

    def __init__(self):
        super().__init__()
        self.chart = ChartWidget(show_toolbar=True, show_legend=True)
        self.chart.setLabel("left", "y")
        self.chart.setLabel("bottom", "x")
        self.chart.setAutofitEnabled(False)
        self.chart.setViewport(-10.0, 10.0, -5.0, 5.0)
        self._ruler = self.chart.addRuler(-3.0, 0.0, 3.0, 0.0, color="#e74c3c", width=2)
        self._ruler_enabled = False
        self._vp_label = QLabel()
        self._vp_label.setStyleSheet("color: #888; font-size: 10px; font-family: monospace;")
        self.chart.onViewportChanged(self._on_viewport_changed)
        self._ruler_label = QLabel("Ruler: off")
        self._ruler_label.setStyleSheet("color: #888; font-size: 10px; font-family: monospace;")
        self._ruler.changed = self._on_ruler_changed
        func_panel = self._build_func_panel()
        controls = self._build_controls()
        info_bar = QHBoxLayout()
        info_bar.addWidget(self._vp_label, 1)
        info_bar.addWidget(self._ruler_label)
        left = QVBoxLayout()
        left.setContentsMargins(4, 4, 4, 4)
        left.setSpacing(4)
        left.addWidget(func_panel)
        left.addLayout(controls)
        left.addLayout(info_bar)
        left.addStretch()
        left_w = QWidget()
        left_w.setFixedWidth(320)
        left_w.setLayout(left)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(left_w)
        root.addWidget(self.chart, 1)
        for expr, color in _PRESET_EXPRESSIONS[:2]:
            self._add_row(expr, color)

    def _build_func_panel(self) -> QWidget:
        grp = QGroupBox("Functions  (use x as variable, math.* available)")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setSpacing(2)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.addStretch()
        scroll.setWidget(self._rows_widget)
        scroll.setMinimumHeight(180)
        add_btn = QPushButton("+ Add function")
        add_btn.clicked.connect(lambda: self._add_row("", "#3498db"))
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        for expr, color in _PRESET_EXPRESSIONS:
            short = expr[:10] + ("…" if len(expr) > 10 else "")
            btn = QPushButton(short)
            btn.setFixedHeight(22)
            btn.setToolTip(expr)
            btn.clicked.connect(lambda checked=False, e=expr, c=color: self._add_row(e, c))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        layout.addWidget(scroll, 1)
        layout.addWidget(add_btn)
        layout.addLayout(preset_row)
        return grp

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._ruler_btn = QPushButton("Ruler: OFF")
        self._ruler_btn.setCheckable(True)
        self._ruler_btn.setFixedWidth(90)
        self._ruler_btn.toggled.connect(self._toggle_ruler)
        zoom_in = QPushButton("Zoom in")
        zoom_out = QPushButton("Zoom out")
        reset_btn = QPushButton("Reset view")
        zoom_in.clicked.connect(lambda: self._zoom(0.6))
        zoom_out.clicked.connect(lambda: self._zoom(1.4))
        reset_btn.clicked.connect(lambda: self.chart.setViewport(-10.0, 10.0, -5.0, 5.0, animated=True))
        for w in (self._ruler_btn, zoom_in, zoom_out, reset_btn):
            row.addWidget(w)
        row.addStretch()
        return row

    def _add_row(self, expr: str, color: str):
        row = _FunctionRow(self.chart, expr, color, self._rows_widget)
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)

    def _toggle_ruler(self, checked: bool):
        self._ruler_enabled = checked
        self._ruler_btn.setText("Ruler: ON" if checked else "Ruler: OFF")
        self._ruler.setVisible(checked)
        if not checked:
            self._ruler_label.setText("Ruler: off")

    def _on_ruler_changed(self):
        if self._ruler_enabled:
            r = self._ruler
            self._ruler_label.setText(
                f"Ruler  d={r.distance:.4g}  dx={r.dx:.4g}  dy={r.dy:.4g}  a={r.angle_deg:.1f}°")

    def _on_viewport_changed(self, x0, x1, y0, y1):
        self._vp_label.setText(
            f"Viewport  x=[{x0:.3g}, {x1:.3g}]  y=[{y0:.3g}, {y1:.3g}]  span_x={x1-x0:.3g}")

    def _zoom(self, factor: float):
        vp = self.chart.viewport
        cx, cy = (vp.x0 + vp.x1) / 2.0, (vp.y0 + vp.y1) / 2.0
        hw_x = (vp.x1 - vp.x0) / 2.0 * factor
        hw_y = (vp.y1 - vp.y0) / 2.0 * factor
        self.chart.setViewport(cx - hw_x, cx + hw_x, cy - hw_y, cy + hw_y, animated=True)


class LiveTab(QWidget):
    """Live streaming data with derivative/integral overlays and error band."""

    def __init__(self):
        super().__init__()
        self._tick = 0
        self._buf = 2000
        self._xs, self._ys1, self._ys2 = [], [], []
        self._sc_x, self._sc_y, self._sc_ey = [], [], []
        self._sc_ann = []
        self._paused = False
        self._phase = 0.0
        self.chart = ChartWidget(show_toolbar=True, show_legend=True,
                                 show_sidebar=True, threaded_fit=True,
                                 anim_duration=120)
        self.chart.setLabel("left", "Amplitude")
        self.chart.setLabel("bottom", "Time, s")
        self.chart.setLabel("right", "Power, W")
        self.line1 = self.chart.plot(label="Signal A", color="#3498db", width=2)
        self.line1.setFillUnder(True, alpha=35)
        self.line2 = self.chart.plot(label="Signal B", color="#2ecc71", width=2)
        self.line2.setStepMode(True)
        self.power = self.chart.plot(label="Power (R)", color="#e74c3c", width=2, right_axis=True)
        self.scatter = self.chart.addScatter(size=9, label="Peaks", color="#f39c12")
        self.scatter.point_clicked.connect(self._on_point_clicked)
        self.fit_a = self.chart.addFit(self.line1, mode_key="poly3",
                                       color="#1a6fa8", label="Fit A", show_formula=True)
        self.hline = self.chart.addLine(y=0.0, color="#9b59b6", width=1)
        self.hline.setVisible(True)
        self._deriv_item = _DerivativeItem(
            self.chart, self.line1, order=1,
            pen=QPen(QColor("#f39c12"), 1, Qt.PenStyle.DashLine), label="dA/dt")
        self._deriv_item.visible = False
        self._chart_items_register(self._deriv_item)
        self._band = _ErrorBandItem(self.chart, color=QColor("#3498db"), alpha=40, label="±σ band")
        self._band.visible = False
        self._chart_items_register(self._band)
        sb = self.chart.sidebar()
        if sb:
            sb.addLabel("Live Controls")
            sb.addSeparator()
            self._btn_pause = sb.addButton("Pause", self._toggle_pause)
            sb.addButton("Clear", self._clear)
            sb.addButton("Autofit", self.chart.autofit)
            sb.addSeparator()
            sb.addLabel("Overlays")
            sb.addButton("d/dt overlay", self._toggle_deriv)
            sb.addButton("±σ band", self._toggle_band)
            sb.addButton("Fill A", lambda: self.line1.setFillUnder(not self.line1.fill_under, 35))
            sb.addButton("Step B", lambda: self.line2.setStepMode(not self.line2.step_mode))
            sb.addButton("Formula", lambda: self.fit_a.setShowFormula(not self.fit_a.show_formula))
            sb.addSeparator()
            sb.addLabel("Zoom lock")
            self._zoom_combo = QComboBox()
            self._zoom_combo.addItems(["Both", "X only", "Y only"])
            self._zoom_combo.currentIndexChanged.connect(
                lambda i: self.chart.setZoomLock(["both", "x", "y"][i]))
            sb.layout().insertWidget(sb.layout().count() - 1, self._zoom_combo)
        self._status = QLabel("Click a peak to inspect  |  Shift+drag to zoom  |  Double-click to set range")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.chart, 1)
        root.addWidget(self._status)
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick_data)
        self.timer.start(1)

    def _chart_items_register(self, item):
        """Register a custom item into the chart's lines list for autofit."""
        if hasattr(self.chart, '_lines'):
            self.chart._lines.append(item)

    def _on_point_clicked(self, x, y, idx):
        self._status.setText(f"Peak selected  x={x:.3f}  y={y:.3f}  idx={idx}")

    def _toggle_pause(self):
        self._paused = not self._paused
        self._btn_pause.setText("Resume" if self._paused else "Pause")

    def _toggle_deriv(self):
        self._deriv_item.visible = not self._deriv_item.visible
        if self._deriv_item.visible and self._xs:
            self._deriv_item.recompute()
        self.chart.update()

    def _toggle_band(self):
        self._band.visible = not self._band.visible
        if self._band.visible and len(self._ys1) >= 6:
            self._band.setFromMeanStd(self._xs, self._ys1, n_sigma=1.0)
        self.chart.update()

    def _clear(self):
        self._tick = 0
        self._xs.clear(); self._ys1.clear(); self._ys2.clear()
        self._sc_x.clear(); self._sc_y.clear()
        self._sc_ey.clear(); self._sc_ann.clear()
        for item, args in ((self.line1, ([], [])), (self.line2, ([], [])), (self.power, ([], []))):
            item.setData(*args)
        self.scatter.setData(x=[], y=[])
        self.scatter.setErrorBars(error_y=[])
        self.scatter.setAnnotations([])
        self._deriv_item._xs, self._deriv_item._ys = [], []
        self._band.clear()

    def _tick_data(self):
        if self._paused:
            return
        self._tick += 1
        self._phase += 0.06
        t = self._tick * 0.05
        while len(self._xs) >= self._buf:
            self._xs.pop(0); self._ys1.pop(0); self._ys2.pop(0)
        self._xs.append(t)
        noise = random.gauss(0, 0.8)
        v1 = 3.5 * math.sin(self._phase) * math.exp(-0.003 * t) + noise
        v2 = 2.0 * math.cos(self._phase * 0.7 + 1.0) + random.gauss(0, 0.4)
        self._ys1.append(v1)
        self._ys2.append(v2)
        if self._tick % 20 == 0:
            while len(self._sc_x) >= self._buf // 10:
                self._sc_x.pop(0); self._sc_y.pop(0)
                self._sc_ey.pop(0); self._sc_ann.pop(0)
            self._sc_x.append(t)
            self._sc_y.append(v1)
            self._sc_ey.append(abs(random.gauss(0.3, 0.1)))
            self._sc_ann.append(f"P{len(self._sc_x)}")
        pwr_xs = self._xs[-50:]
        pwr_ys = [y ** 2 for y in self._ys1[-50:]]
        self.line1.setData(xs=self._xs, ys=self._ys1)
        self.line2.setData(xs=self._xs, ys=self._ys2)
        self.power.setData(xs=pwr_xs, ys=pwr_ys)
        self.scatter.setData(x=self._sc_x, y=self._sc_y)
        self.scatter.setErrorBars(error_y=self._sc_ey)
        self.scatter.setAnnotations(self._sc_ann)
        if self._tick % 10 == 0:
            if self._deriv_item.visible:
                self._deriv_item.recompute()
            if self._band.visible and len(self._ys1) >= 6:
                self._band.setFromMeanStd(self._xs, self._ys1, n_sigma=1.0)


class LogScaleTab(QWidget):
    """Filter frequency response with log axes, scatter measurements, draggable line."""

    def __init__(self):
        super().__init__()
        self.chart = ChartWidget(show_toolbar=True, show_legend=True)
        self.chart.setLabel("left", "Response")
        self.chart.setLabel("bottom", "Frequency, Hz")
        xs_log = [10 ** (i * 0.1) for i in range(0, 40)]
        line_lp = self.chart.plot(label="Low-pass", color="#3498db", width=2)
        ys_lp = [1 / math.sqrt(1 + (x / 100) ** 4) for x in xs_log]
        line_lp.setData(xs=xs_log, ys=ys_lp)
        line_lp.setFillUnder(True, alpha=25)
        line_hp = self.chart.plot(label="High-pass", color="#e74c3c", width=2)
        ys_hp = [(x / 100) ** 2 / math.sqrt(1 + (x / 100) ** 4) for x in xs_log]
        line_hp.setData(xs=xs_log, ys=ys_hp)
        sc = self.chart.addScatter(size=7, label="Measurements", color="#f39c12")
        meas_x = [10 ** (i * 0.3) for i in range(0, 14)]
        meas_y = [1 / math.sqrt(1 + (x / 100) ** 4) + random.gauss(0, 0.02) for x in meas_x]
        meas_ey = [abs(random.gauss(0.015, 0.005)) for _ in meas_x]
        sc.setData(x=meas_x, y=meas_y)
        sc.setErrorBars(error_y=meas_ey)
        vline = self.chart.addLine(x=100.0, color="#9b59b6", width=1)
        vline.setVisible(True)
        ctrl = QGroupBox("Axis scale")
        ctrl_lay = QHBoxLayout(ctrl)
        cb_lx = QCheckBox("Log X")
        cb_ly = QCheckBox("Log Y")
        cb_lx.toggled.connect(lambda v: (self.chart.setLogScale(x=v), self.chart.autofit()))
        cb_ly.toggled.connect(lambda v: (self.chart.setLogScale(y=v), self.chart.autofit()))
        for w in (cb_lx, cb_ly, QLabel("  Shift+drag = rubber-band zoom  |  Drag purple line")):
            ctrl_lay.addWidget(w)
        ctrl_lay.addStretch()
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        root.addWidget(ctrl)
        root.addWidget(self.chart, 1)
        self.chart.autofit()


class BigDataTab(QWidget):
    """100k-point decimation demo with peak detection overlay."""

    def __init__(self):
        super().__init__()
        self.chart = ChartWidget(show_toolbar=True, show_legend=True)
        self.chart.setLabel("left", "Value")
        self.chart.setLabel("bottom", "Index")
        N = 100_000
        xs = list(range(N))
        random.seed(42)
        ys_rw, v = [], 0.0
        for _ in range(N):
            v += random.gauss(0, 1)
            ys_rw.append(v)
        ys_sin = [50 * math.sin(i * 0.001) + 10 * math.sin(i * 0.01) for i in range(N)]
        line_rw = self.chart.plot(label=f"Random walk ({N:,} pts)", color="#3498db", width=1)
        line_rw.setData(xs=xs, ys=ys_rw)
        line_rw.setFillUnder(True, alpha=20)
        line_sin = self.chart.plot(label="Multi-sine", color="#e74c3c", width=1)
        line_sin.setData(xs=xs, ys=ys_sin)
        px, py = peak_find(xs, ys_sin, min_prominence=0.6)
        sc_peaks = self.chart.addScatter(size=7, label=f"Peaks ({len(px)})", color="#f39c12")
        sc_peaks.setData(x=px, y=py)
        info = QLabel(f"{N:,} points per series · {len(px)} peaks detected · scroll to zoom")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: #888; font-size: 11px; padding: 4px;")
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        root.addWidget(info)
        root.addWidget(self.chart, 1)
        self.chart.autofit()


class AnalyticsTab(QWidget):
    """Scatter data with polynomial/exponential fits, Gaussian fit, and new fit modes."""

    def __init__(self):
        super().__init__()
        self.chart = ChartWidget(show_toolbar=True, show_legend=True, show_sidebar=True)
        self.chart.setLabel("left", "y")
        self.chart.setLabel("bottom", "x")
        random.seed(7)
        xs = [i * 0.2 for i in range(60)]
        ys_q = [0.04 * x ** 2 - 1.5 * x + 20 + random.gauss(0, 1.2) for x in xs]
        ys_e = [5 * math.exp(0.05 * x) + random.gauss(0, 0.8) for x in xs]
        sc1 = self.chart.addScatter(size=7, label="Quadratic data", color="#3498db")
        sc1.setData(x=xs, y=ys_q)
        sc1.setErrorBars(error_y=[abs(random.gauss(0.8, 0.2)) for _ in xs])
        sc2 = self.chart.addScatter(size=7, label="Exponential data", color="#e74c3c")
        sc2.setData(x=xs, y=ys_e)
        self.fit1 = self.chart.addFit(sc1, mode_key="poly2", color="#1a6fa8",
                                      label="Poly-2 fit", show_formula=True)
        self.fit2 = self.chart.addFit(sc2, mode_key="linear", color="#c0392b",
                                      label="Linear fit", show_formula=True)
        sb = self.chart.sidebar()
        if sb:
            sb.addLabel("Analytics")
            sb.addSeparator()
            sb.addButton("Show Stats", self.chart._canvas.toggleAnalytics)
            sb.addButton("Set range [10–30]", lambda: (
                self.chart.setRangeSelection(10.0, 30.0),
                self.chart._canvas._set_analytics(True)))
            sb.addButton("Clear range", self.chart.clearRangeSelection)
            sb.addSeparator()
            sb.addLabel("Fit mode (sc1)")
            for key, lbl in (("poly2", "Poly 2"), ("poly3", "Poly 3"), ("gaussian", "Gaussian"),
                               ("logistic", "Logistic"), ("linear", "Linear"), ("spline", "Spline")):
                sb.addButton(lbl, lambda k=key: self._set_fit1(k))
            sb.addSeparator()
            sb.addLabel("Fit mode (sc2)")
            for key, lbl in (("exp", "Exponential"), ("double_exp", "Double Exp"),
                               ("power", "Power"), ("log", "Log")):
                sb.addButton(lbl, lambda k=key: self._set_fit2(k))
        hint = QLabel("Double-click to set range edges · Gaussian/Logistic fits in sidebar")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.chart, 1)
        root.addWidget(hint)
        self.chart.autofit()

    def _set_fit1(self, key):
        self.fit1.setModeKey(key)
        self.chart.update()

    def _set_fit2(self, key):
        self.fit2.setModeKey(key)
        self.chart.update()


class SignalAnalysisTab(QWidget):
    """
    Derivative, cumulative integral, and FFT spectrum of a configurable signal.

    Top chart: raw signal + d/dt overlay + ∫ overlay.
    Bottom chart: frequency spectrum updated on demand.
    """

    def __init__(self):
        super().__init__()
        N = 512
        self._dt = 0.01
        self._N = N
        self._xs = [i * self._dt for i in range(N)]
        self._ys: list[float] = []
        self._signal_kind = "sine"
        self._freq = 3.0
        self._noise = 0.1

        self.chart_time = ChartWidget(show_toolbar=True, show_legend=True)
        self.chart_time.setLabel("left", "Amplitude")
        self.chart_time.setLabel("bottom", "Time, s")

        self.chart_time.setTitle("FFT Spectrum")

        self.chart_freq = ChartWidget(show_toolbar=True, show_legend=True)
        self.chart_freq.setLabel("left", "Amplitude")
        self.chart_freq.setLabel("bottom", "Frequency, Hz")

        self._line_raw = self.chart_time.plot(label="Signal", color="#3498db", width=2)
        self._line_deriv = self.chart_time.plot(label="d/dt", color="#e74c3c", width=1)
        self._line_deriv.setVisible(False)
        self._line_integ = self.chart_time.plot(label="∫ dt", color="#2ecc71", width=1)
        self._line_integ.setVisible(False)

        self._line_spec = self.chart_freq.plot(label="Spectrum", color="#9b59b6", width=2)
        self._line_spec.setFillUnder(True, alpha=30)

        ctrl = self._build_controls()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.chart_time)
        splitter.addWidget(self.chart_freq)
        splitter.setSizes([400, 220])

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        root.addLayout(ctrl)
        root.addWidget(splitter, 1)

        self._rebuild()

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Signal:"))
        self._sig_combo = QComboBox()
        self._sig_combo.addItems(["sine", "square", "sawtooth", "chirp", "damped"])
        self._sig_combo.currentTextChanged.connect(self._on_sig_changed)
        row.addWidget(self._sig_combo)
        row.addWidget(QLabel("Freq:"))
        self._freq_spin = QSpinBox()
        self._freq_spin.setRange(1, 50)
        self._freq_spin.setValue(int(self._freq))
        self._freq_spin.valueChanged.connect(self._on_freq_changed)
        row.addWidget(self._freq_spin)
        row.addWidget(QLabel("Noise:"))
        self._noise_sl = QSlider(Qt.Orientation.Horizontal)
        self._noise_sl.setRange(0, 100)
        self._noise_sl.setValue(10)
        self._noise_sl.setFixedWidth(80)
        self._noise_sl.valueChanged.connect(self._on_noise_changed)
        row.addWidget(self._noise_sl)
        cb_d = QCheckBox("d/dt")
        cb_d.toggled.connect(self._toggle_deriv)
        cb_i = QCheckBox("∫ dt")
        cb_i.toggled.connect(self._toggle_integ)
        row.addWidget(cb_d)
        row.addWidget(cb_i)
        row.addStretch()
        return row

    def _generate(self) -> list[float]:
        random.seed(0)
        ys = []
        for i, t in enumerate(self._xs):
            w = 2 * math.pi * self._freq * t
            if self._signal_kind == "sine":
                v = math.sin(w)
            elif self._signal_kind == "square":
                v = 1.0 if math.sin(w) >= 0 else -1.0
            elif self._signal_kind == "sawtooth":
                v = 2 * ((self._freq * t) % 1.0) - 1.0
            elif self._signal_kind == "chirp":
                inst_freq = self._freq * (1 + t / self._xs[-1])
                v = math.sin(2 * math.pi * inst_freq * t)
            elif self._signal_kind == "damped":
                v = math.exp(-t * 1.5) * math.sin(w)
            else:
                v = 0.0
            ys.append(v + random.gauss(0, self._noise))
        return ys

    def _rebuild(self):
        self._ys = self._generate()
        self._line_raw.setData(xs=self._xs, ys=self._ys)
        if self._line_deriv.visible:
            dxs, dys = derivative(self._xs, self._ys)
            self._line_deriv.setData(xs=dxs, ys=dys)
        if self._line_integ.visible:
            ixs, iys = cumulative_integral(self._xs, self._ys)
            self._line_integ.setData(xs=ixs, ys=iys)
        freqs, amps = fft_spectrum_numpy(self._xs, self._ys)
        self._line_spec.setData(xs=freqs, ys=amps)
        self.chart_time.autofit()
        self.chart_freq.autofit()

    def _on_sig_changed(self, kind: str):
        self._signal_kind = kind
        self._rebuild()

    def _on_freq_changed(self, v: int):
        self._freq = float(v)
        self._rebuild()

    def _on_noise_changed(self, v: int):
        self._noise = v / 200.0
        self._rebuild()

    def _toggle_deriv(self, checked: bool):
        self._line_deriv.setVisible(checked)
        if checked and self._ys:
            dxs, dys = derivative(self._xs, self._ys)
            self._line_deriv.setData(xs=dxs, ys=dys)
        self.chart_time.autofit()

    def _toggle_integ(self, checked: bool):
        self._line_integ.setVisible(checked)
        if checked and self._ys:
            ixs, iys = cumulative_integral(self._xs, self._ys)
            self._line_integ.setData(xs=ixs, ys=iys)
        self.chart_time.autofit()


class StatisticsTab(QWidget):
    """
    Histogram, autocorrelation, normalization, and peak detection on a generated dataset.

    Left: histogram with configurable bins and normalization toggle.
    Right: ACF (autocorrelation function) and normalized signal overlaid.
    """

    def __init__(self):
        super().__init__()
        random.seed(99)
        N = 800
        self._xs_t = [i * 0.05 for i in range(N)]
        self._data_bimodal = (
            [random.gauss(-2.0, 0.7) for _ in range(N // 2)] +
            [random.gauss(2.5, 1.1) for _ in range(N // 2)]
        )
        self._data_signal = [
            math.sin(2 * math.pi * 0.3 * t) + random.gauss(0, 0.4)
            for t in self._xs_t
        ]

        self.chart_hist = ChartWidget(show_toolbar=True, show_legend=True)
        self.chart_hist.setLabel("left", "Count")
        self.chart_hist.setLabel("bottom", "Value")

        self.chart_acf = ChartWidget(show_toolbar=True, show_legend=True)
        self.chart_acf.setLabel("left", "ACF / Amplitude")
        self.chart_acf.setLabel("bottom", "Lag / Time")

        self._bins = 30
        self._normalize_hist = False
        self._line_hist = self.chart_hist.plot(label="Bimodal dist.", color="#3498db", width=2)
        self._line_hist.setFillUnder(True, alpha=40)

        self._line_acf = self.chart_acf.plot(label="ACF (signal)", color="#e74c3c", width=2)
        self._line_acf.setFillUnder(True, alpha=25)
        self._line_norm_minmax = self.chart_acf.plot(label="Signal norm. (minmax)", color="#3498db", width=1)
        self._line_norm_zscore = self.chart_acf.plot(label="Signal norm. (zscore)", color="#2ecc71", width=1)
        self._line_norm_minmax.setVisible(False)
        self._line_norm_zscore.setVisible(False)

        sc_peaks = self.chart_acf.addScatter(size=8, label="Peaks", color="#f39c12")
        px, py = peak_find(self._xs_t, self._data_signal, min_prominence=0.3)
        sc_peaks.setData(x=px, y=py)

        ctrl = self._build_controls()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.chart_hist)
        splitter.addWidget(self.chart_acf)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        root.addLayout(ctrl)
        root.addWidget(splitter, 1)

        self._rebuild_hist()
        self._rebuild_acf()

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Bins:"))
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(5, 100)
        self._bins_spin.setValue(self._bins)
        self._bins_spin.valueChanged.connect(self._on_bins_changed)
        row.addWidget(self._bins_spin)
        self._norm_cb = QCheckBox("Normalize histogram")
        self._norm_cb.toggled.connect(self._on_norm_toggled)
        row.addWidget(self._norm_cb)
        cb_mm = QCheckBox("norm. minmax")
        cb_mm.toggled.connect(self._toggle_norm_minmax)
        cb_zs = QCheckBox("norm. zscore")
        cb_zs.toggled.connect(self._toggle_norm_zscore)
        row.addWidget(cb_mm)
        row.addWidget(cb_zs)
        row.addStretch()
        return row

    def _rebuild_hist(self):
        centers, counts = histogram(self._data_bimodal, self._bins)
        if self._normalize_hist:
            total = sum(counts)
            counts = [c / total if total > 0 else 0.0 for c in counts]
            self.chart_hist.setLabel("left", "Probability density")
        else:
            self.chart_hist.setLabel("left", "Count")
        self._line_hist.setData(xs=centers, ys=counts)
        self.chart_hist.autofit()

    def _rebuild_acf(self):
        lags, acf = autocorrelation(self._data_signal, max_lag=100)
        self._line_acf.setData(xs=[float(l) for l in lags], ys=acf)
        self.chart_acf.autofit()

    def _on_bins_changed(self, v: int):
        self._bins = v
        self._rebuild_hist()

    def _on_norm_toggled(self, checked: bool):
        self._normalize_hist = checked
        self._rebuild_hist()

    def _toggle_norm_minmax(self, checked: bool):
        self._line_norm_minmax.setVisible(checked)
        if checked:
            nm = normalize(self._data_signal, "minmax")
            self._line_norm_minmax.setData(xs=self._xs_t, ys=nm)
        self.chart_acf.autofit()

    def _toggle_norm_zscore(self, checked: bool):
        self._line_norm_zscore.setVisible(checked)
        if checked:
            nz = normalize(self._data_signal, "zscore")
            self._line_norm_zscore.setData(xs=self._xs_t, ys=nz)
        self.chart_acf.autofit()


class DemoWindow(QMainWindow):
    """Main window hosting all demo tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChartWidget · Feature Demo")
        self.resize(1280, 760)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(GraphingCalculatorTab(), "Graphing Calculator")
        tabs.addTab(LiveTab(), "Live stream")
        tabs.addTab(LogScaleTab(), "Log scale / Filter")
        tabs.addTab(BigDataTab(), "100k points")
        tabs.addTab(AnalyticsTab(), "Analytics & Fit")
        tabs.addTab(SignalAnalysisTab(), "Signal Analysis")
        tabs.addTab(StatisticsTab(), "Statistics")
        bar = QStatusBar()
        bar.showMessage(
            "Graphing Calc: type expressions  ·  Signal Analysis: d/dt & FFT  ·  "
            "Statistics: histogram & ACF  ·  Pan: drag  ·  Zoom: scroll  ·  Right-click: menu")
        self.setStatusBar(bar)
        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DemoWindow()
    win.show()
    sys.exit(app.exec_())