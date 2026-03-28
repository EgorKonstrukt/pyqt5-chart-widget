from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING
from PyQt5.QtCore import QThread, pyqtSignal
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
        self.raw_visible = True
    def setData(self, xs=None, ys=None):
        self.xs = list(xs) if xs is not None else []
        self.ys = list(ys) if ys is not None else []
        self._chart._schedule_autofit()
    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()
    def setRawVisible(self, v: bool):
        self.raw_visible = v
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
        self.raw_visible = True
    def setData(self, x=None, y=None, **_):
        self.xs = list(x) if x is not None else []
        self.ys = list(y) if y is not None else []
        self._chart._schedule_autofit()
    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()
    def setRawVisible(self, v: bool):
        self.raw_visible = v
        self._chart.update()
    def setLabel(self, label: str):
        self.label = label
        self._chart.update()


class _FitWorker(QThread):
    result_ready = pyqtSignal(object, list, list)
    def __init__(self, fit_item: "_FitItem", x_lo: float, x_hi: float, n_pts: int):
        super().__init__()
        self._fit_item = fit_item
        self._x_lo = x_lo
        self._x_hi = x_hi
        self._n_pts = n_pts
    def run(self):
        from .math_utils import get_fit_mode, linspace
        mode = get_fit_mode(self._fit_item.mode_key)
        if mode is None or not self._fit_item.source.xs:
            self.result_ready.emit(self._fit_item, [], [])
            return
        x_eval = linspace(self._x_lo, self._x_hi, self._n_pts)
        result = mode.evaluate(list(self._fit_item.source.xs), list(self._fit_item.source.ys), x_eval)
        if result is None:
            self.result_ready.emit(self._fit_item, [], [])
        else:
            self.result_ready.emit(self._fit_item, x_eval, result)


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
        self._worker: Optional[_FitWorker] = None
        self._pending_range: Optional[Tuple[float, float, int]] = None
    def _recompute(self, x_range_lo: float, x_range_hi: float,
                   n_pts: int = 400, threaded: bool = False):
        if not threaded:
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
            return
        self._pending_range = (x_range_lo, x_range_hi, n_pts)
        if self._worker is None or not self._worker.isRunning():
            self._start_worker(x_range_lo, x_range_hi, n_pts)
    def _start_worker(self, x_lo: float, x_hi: float, n_pts: int):
        self._worker = _FitWorker(self, x_lo, x_hi, n_pts)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.start()
    def _on_worker_result(self, fit_item: "_FitItem", xs: List[float], ys: List[float]):
        if fit_item is not self:
            return
        self._xs = xs
        self._ys = ys
        self._chart._canvas.update()
        if self._pending_range is not None:
            lo, hi, n = self._pending_range
            if (self._worker is not None and
                    (abs(lo - self._worker._x_lo) > 1e-10 or
                     abs(hi - self._worker._x_hi) > 1e-10)):
                self._start_worker(lo, hi, n)
            else:
                self._pending_range = None
    def getData(self, x_lo: Optional[float] = None, x_hi: Optional[float] = None,
                n_pts: int = 400) -> Tuple[List[float], List[float]]:
        lo = x_lo if x_lo is not None else (min(self.source.xs) if self.source.xs else 0.0)
        hi = x_hi if x_hi is not None else (max(self.source.xs) if self.source.xs else 1.0)
        self._recompute(lo, hi, n_pts, threaded=False)
        return list(self._xs), list(self._ys)
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