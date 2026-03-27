from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen

if TYPE_CHECKING:
    from .chart_widget import ChartWidget


class _InfLine:
    def __init__(self, chart: "ChartWidget", horizontal: bool, value: float, pen: QPen):
        self._chart = chart
        self.horizontal = horizontal
        self.value = value
        self.pen = pen
        self.visible = False

    def setValue(self, v: float):
        self.value = v
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()


class _LineItem:
    def __init__(self, chart: "ChartWidget", pen: QPen, label: str = ""):
        self._chart = chart
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.pen = pen
        self.label = label
        self.visible = True

    def setData(self, xs=None, ys=None):
        self.xs = list(xs) if xs is not None else []
        self.ys = list(ys) if ys is not None else []
        self._chart._schedule_autofit()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()


class _ScatterItem:
    def __init__(self, chart: "ChartWidget", size: int, color: QColor, label: str = ""):
        self._chart = chart
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.size = size
        self.color = color
        self.label = label
        self.visible = True

    def setData(self, x=None, y=None, **_):
        self.xs = list(x) if x is not None else []
        self.ys = list(y) if y is not None else []
        self._chart._schedule_autofit()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()


class _FitItem:
    def __init__(self, chart: "ChartWidget", source: "_LineItem | _ScatterItem",
                 mode_key: str, pen: QPen, label: str = ""):
        self._chart = chart
        self.source = source
        self.mode_key = mode_key
        self.pen = pen
        self.label = label
        self.visible = True
        self._xs: List[float] = []
        self._ys: List[float] = []

    def _recompute(self, x_range_lo: float, x_range_hi: float, n_pts: int = 400):
        from .math_utils import get_fit_mode, linspace
        mode = get_fit_mode(self.mode_key)
        if mode is None or not self.source.xs:
            self._xs = []
            self._ys = []
            return
        x_eval = linspace(x_range_lo, x_range_hi, n_pts)
        result = mode.evaluate(list(self.source.xs), list(self.source.ys), x_eval)
        if result is None:
            self._xs = []
            self._ys = []
        else:
            self._xs = x_eval
            self._ys = result

    def setModeKey(self, key: str):
        self.mode_key = key
        self._chart.update()

    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()

    def setLabel(self, label: str):
        self.label = label
        self._chart.update()

    @property
    def xs(self) -> List[float]:
        return self._xs

    @property
    def ys(self) -> List[float]:
        return self._ys
