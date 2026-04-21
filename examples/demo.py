import sys
import math
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
                              QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
                              QCheckBox, QSlider, QSpinBox, QPushButton,
                              QComboBox, QStatusBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from pyqt5_chart_widget import (ChartWidget, FitMode, register_fit_mode,
                                set_palette, update_strings)


def _damped_fit(x_pts, y_pts, x_eval):
    if not x_pts:
        return [0.0] * len(x_eval)
    amp = max(abs(y) for y in y_pts) if y_pts else 1.0
    freq = 0.4
    decay = 0.05
    return [amp * math.exp(-decay * xi) * math.cos(freq * xi) for xi in x_eval]


register_fit_mode(FitMode("damped", "Damped cosine", _damped_fit, min_points=3))
update_strings({"chart_widget.btn_fit": "Fit"})


class LiveTab(QWidget):
    def __init__(self):
        super().__init__()
        self._tick = 0
        self._buf = 200
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
        self.timer.start(20)

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
        self.resize(1100, 680)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(LiveTab(), "Live stream")
        tabs.addTab(LogScaleTab(), "Log scale / Filter")
        tabs.addTab(BigDataTab(), "100k points")
        tabs.addTab(AnalyticsTab(), "Analytics & Fit")

        bar = QStatusBar()
        bar.showMessage("Shift+drag → rubber-band zoom  ·  Double-click → range edges  "
                        "·  Drag dashed lines  ·  Click scatter points  ·  Right-click for context menu")
        self.setStatusBar(bar)
        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DemoWindow()
    win.show()
    sys.exit(app.exec_())