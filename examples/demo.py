from __future__ import annotations
import sys
import math
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
                              QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
                              QCheckBox, QSlider, QSpinBox, QPushButton,
                              QComboBox, QStatusBar, QLineEdit, QToolButton,
                              QScrollArea, QFrame, QColorDialog)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QPen
from pyqt5_chart_widget import (ChartWidget, FitMode, register_fit_mode,
                                set_palette, update_strings,
                                _FunctionItem, _RulerItem)

def _damped_fit(x_pts, y_pts, x_eval):
    if not x_pts:
        return [0.0] * len(x_eval)
    amp = max(abs(y) for y in y_pts) if y_pts else 1.0
    freq = 0.4
    decay = 0.05
    return [amp * math.exp(-decay * xi) * math.cos(freq * xi) for xi in x_eval]


register_fit_mode(FitMode("damped", "Damped cosine", _damped_fit, min_points=3))
update_strings({"chart_widget.btn_fit": "Fit"})


_SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_SAFE_MATH["abs"] = abs


def _compile_expression(expr: str):
    """
    Compile a user-supplied mathematical expression string into a callable
    f(xs) -> List[float | None].

    The expression is evaluated with x as the independent variable and the
    full ``math`` module namespace available (sin, cos, pi, e, etc.).
    Division by zero and other evaluation errors at individual points return
    None so that gaps are rendered instead of crashing.

    Returns a callable, or None if the expression fails to compile.
    """
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
    """
    A single row in the graphing calculator: expression input, colour button,
    visibility toggle and remove button.
    """

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
        row.addWidget(self._color_btn)
        row.addWidget(self._edit, 1)
        row.addWidget(self._status)
        row.addWidget(self._vis_btn)
        row.addWidget(self._remove_btn)

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
    A graphing calculator demo.

    All function evaluation happens here in application code using Python's
    math module.  The library receives a plain callable and knows nothing about
    expressions; it only evaluates the function over the current viewport and
    renders the result.

    Demonstrates:
    - addFunction() for infinite/viewport-adaptive curves
    - addRuler() with draggable endpoints and live distance readout
    - onViewportChanged() for viewport data API
    - setViewport() for programmatic zoom
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

        self._rows: list[_FunctionRow] = []

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
        self._rows.append(row)

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
                f"Ruler  d={r.distance:.4g}  dx={r.dx:.4g}  dy={r.dy:.4g}  "
                f"a={r.angle_deg:.1f}°")

    def _on_viewport_changed(self, x0: float, x1: float, y0: float, y1: float):
        self._vp_label.setText(
            f"Viewport  x=[{x0:.3g}, {x1:.3g}]  y=[{y0:.3g}, {y1:.3g}]  "
            f"span_x={x1-x0:.3g}")

    def _zoom(self, factor: float):
        vp = self.chart.viewport
        cx = (vp.x0 + vp.x1) / 2.0
        cy = (vp.y0 + vp.y1) / 2.0
        hw_x = (vp.x1 - vp.x0) / 2.0 * factor
        hw_y = (vp.y1 - vp.y0) / 2.0 * factor
        self.chart.setViewport(cx - hw_x, cx + hw_x, cy - hw_y, cy + hw_y, animated=True)


class LiveTab(QWidget):
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

        self.power = self.chart.plot(label="Power (R)", color="#e74c3c",
                                     width=2, right_axis=True)

        self.scatter = self.chart.addScatter(size=9, label="Peaks", color="#f39c12")
        self.scatter.point_clicked.connect(self._on_point_clicked)

        self.fit_a = self.chart.addFit(self.line1, mode_key="poly3",
                                       color="#1a6fa8", label="Fit A",
                                       show_formula=True)

        self.hline = self.chart.addLine(y=0.0, color="#9b59b6", width=1)
        self.hline.setVisible(True)

        sb = self.chart.sidebar()
        if sb:
            sb.addLabel("Live Controls")
            sb.addSeparator()
            self._btn_pause = sb.addButton("Pause", self._toggle_pause)
            sb.addButton("Clear", self._clear)
            sb.addButton("Autofit", self.chart.autofit)
            sb.addSeparator()
            sb.addLabel("Display")
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

    def _on_point_clicked(self, x, y, idx):
        self._status.setText(f"Peak selected  x={x:.3f}  y={y:.3f}  idx={idx}")

    def _toggle_pause(self):
        self._paused = not self._paused
        self._btn_pause.setText("Resume" if self._paused else "Pause")

    def _clear(self):
        self._tick = 0
        self._xs.clear(); self._ys1.clear(); self._ys2.clear()
        self._sc_x.clear(); self._sc_y.clear()
        self._sc_ey.clear(); self._sc_ann.clear()
        for item, args in ((self.line1, ([], [])), (self.line2, ([], [])),
                            (self.power, ([], []))):
            item.setData(*args)
        self.scatter.setData(x=[], y=[])
        self.scatter.setErrorBars(error_y=[])
        self.scatter.setAnnotations([])

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


class LogScaleTab(QWidget):
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
        cb_lx.setChecked(False)
        cb_ly.setChecked(False)
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
    def __init__(self):
        super().__init__()
        self.chart = ChartWidget(show_toolbar=True, show_legend=True)
        self.chart.setLabel("left", "Value")
        self.chart.setLabel("bottom", "Index")

        N = 100_000
        xs = list(range(N))
        random.seed(42)
        ys_rw = []
        v = 0.0
        for _ in range(N):
            v += random.gauss(0, 1)
            ys_rw.append(v)

        ys_sin = [50 * math.sin(i * 0.001) + 10 * math.sin(i * 0.01) for i in range(N)]

        line_rw = self.chart.plot(label=f"Random walk ({N:,} pts)", color="#3498db", width=1)
        line_rw.setData(xs=xs, ys=ys_rw)
        line_rw.setFillUnder(True, alpha=20)

        line_sin = self.chart.plot(label="Multi-sine", color="#e74c3c", width=1)
        line_sin.setData(xs=xs, ys=ys_sin)

        info = QLabel(f"{N:,} points per series · scroll to zoom")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: #888; font-size: 11px; padding: 4px;")

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        root.addWidget(info)
        root.addWidget(self.chart, 1)
        self.chart.autofit()


class AnalyticsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.chart = ChartWidget(show_toolbar=True, show_legend=True,
                                 show_sidebar=True)
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

        fit1 = self.chart.addFit(sc1, mode_key="poly2", color="#1a6fa8",
                                  label="Poly-2 fit", show_formula=True)
        fit2 = self.chart.addFit(sc2, mode_key="linear", color="#c0392b",
                                  label="Linear fit", show_formula=True)

        sb = self.chart.sidebar()
        if sb:
            sb.addLabel("Analytics")
            sb.addSeparator()
            sb.addButton("Show Stats", self.chart._canvas.toggleAnalytics)
            sb.addButton("Set range\n[10 – 30]", lambda: (
                self.chart.setRangeSelection(10.0, 30.0),
                self.chart._canvas._set_analytics(True)))
            sb.addButton("Clear range", self.chart.clearRangeSelection)
            sb.addSeparator()
            sb.addLabel("Fit mode")
            for key, lbl in (("poly2", "Poly 2"), ("poly3", "Poly 3"),
                               ("linear", "Linear"), ("spline", "Spline")):
                sb.addButton(lbl, lambda k=key: self._set_fit(k))

        hint = QLabel("Double-click to set range edges · Stats panel shows integral & range stats")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.chart, 1)
        root.addWidget(hint)
        self.chart.autofit()

    def _set_fit(self, key):
        for fit in self.chart.fits:
            fit.setModeKey(key)
        self.chart.update()


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChartWidget · Feature Demo")
        self.resize(1200, 720)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(GraphingCalculatorTab(), "Graphing Calculator")
        tabs.addTab(LiveTab(), "Live stream")
        tabs.addTab(LogScaleTab(), "Log scale / Filter")
        tabs.addTab(BigDataTab(), "100k points")
        tabs.addTab(AnalyticsTab(), "Analytics & Fit")

        bar = QStatusBar()
        bar.showMessage(
            "Graphing Calc: type expressions using x  ·  Ruler: toggle ON then drag handles  ·  "
            "Pan: drag  ·  Zoom: scroll  ·  Rubber-band: Shift+drag  ·  Right-click for menu")
        self.setStatusBar(bar)
        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DemoWindow()
    win.show()
    sys.exit(app.exec_())