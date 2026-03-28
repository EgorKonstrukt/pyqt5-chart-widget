import sys
import math
import random
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from pyqt5_chart_widget import (ChartWidget, FitMode, register_fit_mode,
                                set_palette, update_strings)


def _sinc_fit(x_pts, y_pts, x_eval):
    return [math.sin(xi * 0.01) * 50 + 50 for xi in x_eval]


register_fit_mode(FitMode("sinc_demo", "Sinc (custom)", _sinc_fit, min_points=1))

update_strings({"chart_widget.btn_fit": "⊞ Fit"})


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._tick = 0
        self._max_points = 100
        self._xs = []
        self._ys1 = []
        self._ys2 = []
        self._scatter_x = []
        self._scatter_y = []

        self.chart = ChartWidget(show_toolbar=True, show_legend=True, show_sidebar=True)
        self.chart.setLabel("left", "Flow, g/min")
        self.chart.setLabel("bottom", "Time, s")
        self.chart.autofit()

        self.line1 = self.chart.plot(label="Pump A", width=2, color="#3498db")
        self.line2 = self.chart.plot(label="Pump B", width=3, color="#2ecc71")
        self.scatter1 = self.chart.addScatter(size=8, label="Samples", color="#f39c12")

        self.fit1 = self.chart.addFit(self.line1, mode_key="poly3", dashed=True, label="Fit A")
        self.fit2 = self.chart.addFit(self.line2, mode_key="pchip", dashed=True, label="Fit B")

        self.hline = self.chart.addLine(y=80.0, color="#e74c3c", width=2)

        sb = self.chart.sidebar()
        if sb:
            sb.addLabel("Controls")
            sb.addSeparator()
            sb.addButton("Pause/Resume", self._toggle_timer, "Toggle data stream")
            sb.addButton("Clear Data", self._clear_data, "Reset all data")
            sb.addButton("Toggle A", lambda: self.line1.setVisible(not self.line1.visible), "Show/hide Pump A")
            sb.addButton("Toggle B", lambda: self.line2.setVisible(not self.line2.visible), "Show/hide Pump B")

        self.setCentralWidget(self.chart)
        self.resize(1000, 600)
        self.setWindowTitle("ChartWidget Dynamic Demo")

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_data)
        self.timer.start(16)

    def _toggle_timer(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(50)

    def _clear_data(self):
        self._tick = 0
        self._xs.clear()
        self._ys1.clear()
        self._ys2.clear()
        self._scatter_x.clear()
        self._scatter_y.clear()
        self.line1.setData([], [])
        self.line2.setData([], [])
        self.scatter1.setData([], [])

    def _update_data(self):
        self._tick += 1
        t = self._tick * 0.1

        if len(self._xs) >= self._max_points:
            self._xs.pop(0)
            self._ys1.pop(0)
            self._ys2.pop(0)
            if self._scatter_x:
                self._scatter_x.pop(0)
                self._scatter_y.pop(0)

        self._xs.append(t)

        noise1 = random.gauss(0, 2)
        noise2 = random.gauss(0, 3)

        val1 = 50 + 20 * math.sin(t * 0.5) + 0.5 * t + noise1
        val2 = 50 + 15 * math.cos(t * 0.3) + 0.3 * t + noise2

        self._ys1.append(val1)
        self._ys2.append(val2)

        if self._tick % 5 == 0:
            self._scatter_x.append(t)
            self._scatter_y.append(val1 + random.gauss(0, 1))

        self.line1.setData(xs=self._xs, ys=self._ys1)
        self.line2.setData(xs=self._xs, ys=self._ys2)
        self.scatter1.setData(x=self._scatter_x, y=self._scatter_y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    sys.exit(app.exec_())