from __future__ import annotations
import math
import statistics
from bisect import bisect_left
from typing import List, Optional, Tuple, TYPE_CHECKING
from PyQt5.QtWidgets import QWidget, QSizePolicy, QMenu, QAction, QActionGroup
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                          QFontMetrics, QPainterPath, QPixmap, QWheelEvent, QMouseEvent)
from .math_utils import (nice_ticks, nice_log_ticks, to_log, decimated,
                          fmt, get_fit_modes, trapezoid_integral)
from .items import _LineItem, _ScatterItem, _FitItem, _InfLine
from .i18n import tr
if TYPE_CHECKING:
    from .chart_widget import ChartWidget

try:
    from ._cy_utils import nearest_on_segments_cy as _cy_nearest, decimated_to_screen_cy as _cy_dec_screen
    _CY = True
    # print(_CY)
except ImportError:
    _CY = False
    # print(_CY)

_ML, _MT, _MR, _MB = 58, 14, 20, 40
_MR2 = 52
_ZOOM_FACTOR = 1.15
_ZOOM_MIN_SPAN = 1e-10
_ZOOM_MAX_SPAN = 1e15
_SNAP_RADIUS_PX = 40
_TANGENT_HALF_FRAC = 0.18
_ANALYTICS_PAD = 8
_ANALYTICS_ROW_H = 17
_ANALYTICS_MAX_SERIES = 8
_TOOLTIP_MARGIN = 14
_SNAP_DOT_R = 5.0
_LEGEND_PAD = 8
_LEGEND_SWATCH = 12
_LATEST_TAG_PAD = 3
_LATEST_TAG_ROUND = 3
_COORD_CLAMP = 1e6
_DECIMATE_THRESHOLD = 2000
_INFLINE_HIT_PX = 6
_RUBBERBAND_MIN_PX = 4
_RANGE_SEL_ALPHA = 30

_GRID_PRESETS_X = [("chart_widget.ctx_sparse", 120), ("chart_widget.ctx_normal", 80), ("chart_widget.ctx_dense", 50)]
_GRID_PRESETS_Y = [("chart_widget.ctx_sparse", 100), ("chart_widget.ctx_normal", 60), ("chart_widget.ctx_dense", 40)]


class _PlotCanvas(QWidget):
    def __init__(self, chart: "ChartWidget"):
        super().__init__(chart)
        self._chart = chart
        self._pan_start: Optional[QPointF] = None
        self._pan_vx0 = 0.0
        self._pan_vy0 = 0.0
        self._mouse_pos: Optional[QPointF] = None
        self._show_analytics = False
        self._show_latest = False
        self._rb_start: Optional[QPointF] = None
        self._rb_end: Optional[QPointF] = None
        self._rb_active = False
        self._dragging_infline: Optional[_InfLine] = None
        self._zoom_lock: str = "both"
        self._range_sel_x: Optional[Tuple[float, float]] = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self._nearest_cache_key: Optional[tuple] = None
        self._nearest_cache_result = None

    def toggleAnalytics(self):
        self._show_analytics = not self._show_analytics
        self.update()

    def toggleLatestPoint(self):
        self._show_latest = not self._show_latest
        self.update()

    def _set_analytics(self, v: bool):
        self._show_analytics = v
        self.update()

    def _set_latest(self, v: bool):
        self._show_latest = v
        self.update()

    def setZoomLock(self, lock: str):
        self._zoom_lock = lock

    def clearRangeSelection(self):
        self._range_sel_x = None
        self.update()

    def _has_right_axis(self) -> bool:
        return bool(self._chart.lines_r2 or self._chart.scatters_r2)

    def _plot_rect(self) -> QRect:
        ml = _ML + (4 if self._chart.label_left else 0)
        mr = (_MR2 if self._has_right_axis() else _MR) + (4 if self._chart.label_right else 0)
        return QRect(ml, _MT, max(1, self.width() - ml - mr), max(1, self.height() - _MT - _MB))

    def _to_pt(self, xv, yv, x0, dx, y0, dy, pr) -> QPointF:
        px = pr.left() + (xv - x0) / dx * pr.width()
        py = pr.bottom() - (yv - y0) / dy * pr.height()
        return QPointF(
            max(-_COORD_CLAMP, min(_COORD_CLAMP, px)),
            max(-_COORD_CLAMP, min(_COORD_CLAMP, py)),
        )

    def _to_pt_log(self, xv, yv, x0, dx, y0, dy, pr,
                   log_x: bool, log_y: bool) -> QPointF:
        lx = to_log(xv) if log_x else xv
        ly = to_log(yv) if log_y else yv
        return self._to_pt(lx, ly, x0, dx, y0, dy, pr)

    def _screen_to_data(self, sx: float, sy: float, x0, dx, y0, dy, pr,
                        log_x: bool, log_y: bool) -> Tuple[float, float]:
        lx = x0 + (sx - pr.left()) / pr.width() * dx
        ly = y0 + (pr.bottom() - sy) / pr.height() * dy
        xv = 10 ** lx if log_x else lx
        yv = 10 ** ly if log_y else ly
        return xv, yv

    def _nearest_on_segments(self, mouse: QPointF, xs: List[float], ys: List[float],
                              pr, x0, dx, y0, dy,
                              log_x: bool, log_y: bool) -> Optional[Tuple[float, float, float]]:
        if not xs:
            return None
        if len(xs) == 1:
            pt = self._to_pt_log(xs[0], ys[0], x0, dx, y0, dy, pr, log_x, log_y)
            return (xs[0], ys[0], math.hypot(pt.x() - mouse.x(), pt.y() - mouse.y()))
        best_d = float("inf")
        best_xi, best_yi = xs[0], ys[0]
        for i in range(len(xs) - 1):
            pt0 = self._to_pt_log(xs[i], ys[i], x0, dx, y0, dy, pr, log_x, log_y)
            pt1 = self._to_pt_log(xs[i + 1], ys[i + 1], x0, dx, y0, dy, pr, log_x, log_y)
            seg_x = pt1.x() - pt0.x()
            seg_y = pt1.y() - pt0.y()
            seg_len2 = seg_x * seg_x + seg_y * seg_y
            t = 0.0 if seg_len2 < 1e-10 else max(0.0, min(1.0,
                ((mouse.x() - pt0.x()) * seg_x + (mouse.y() - pt0.y()) * seg_y) / seg_len2))
            nx = pt0.x() + t * seg_x
            ny = pt0.y() + t * seg_y
            d = math.hypot(nx - mouse.x(), ny - mouse.y())
            if d < best_d:
                best_d = d
                best_xi = xs[i] + t * (xs[i + 1] - xs[i])
                best_yi = ys[i] + t * (ys[i + 1] - ys[i])
        return (best_xi, best_yi, best_d)

    def _find_nearest(self, mouse, pr, x0, dx, y0, dy):
        key = (int(mouse.x()), int(mouse.y()), x0, dx, y0, dy)
        if self._nearest_cache_key == key:
            return self._nearest_cache_result
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        best_d = float("inf")
        best = None
        pl, pb, pw, ph = float(pr.left()), float(pr.bottom()), float(pr.width()), float(pr.height())
        for item in c.scatters:
            if not item.visible:
                continue
            for xi, yi in zip(item.xs, item.ys):
                pt = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
                d = math.hypot(pt.x() - mouse.x(), pt.y() - mouse.y())
                if d < best_d:
                    best_d = d
                    best = (xi, yi, d, item)
        for item in c.lines:
            if not item.visible or not item.xs:
                continue
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            if _CY:
                r = _cy_nearest(mouse.x(), mouse.y(), dxs, dys, x0, dx, y0, dy, pl, pb, pw, ph, log_x, log_y)
            else:
                r = self._nearest_on_segments(mouse, dxs, dys, pr, x0, dx, y0, dy, log_x, log_y)
            if r and r[2] < best_d:
                best_d = r[2]
                best = (r[0], r[1], r[2], item)
        for item in c.fits:
            if not item.visible or not item.xs:
                continue
            if _CY:
                r = _cy_nearest(mouse.x(), mouse.y(), item.xs, item.ys, x0, dx, y0, dy, pl, pb, pw, ph, log_x, log_y)
            else:
                r = self._nearest_on_segments(mouse, item.xs, item.ys, pr, x0, dx, y0, dy, log_x, log_y)
            if r and r[2] < best_d:
                best_d = r[2]
                best = (r[0], r[1], r[2], item)
        result = best if best and best[2] <= _SNAP_RADIUS_PX else None
        self._nearest_cache_key = key
        self._nearest_cache_result = result
        return result

    def _find_infline_at(self, pos: QPointF, pr, x0, dx, y0, dy) -> Optional[_InfLine]:
        c = self._chart
        for ln in c.inflines:
            if not ln.visible or not ln.draggable:
                continue
            if ln.horizontal:
                sy = pr.bottom() - (ln.value - y0) / dy * pr.height()
                if abs(pos.y() - sy) <= _INFLINE_HIT_PX and pr.left() <= pos.x() <= pr.right():
                    return ln
            else:
                sx = pr.left() + (ln.value - x0) / dx * pr.width()
                if abs(pos.x() - sx) <= _INFLINE_HIT_PX and pr.top() <= pos.y() <= pr.bottom():
                    return ln
        return None

    def _find_scatter_point_at(self, pos: QPointF, pr, x0, dx, y0, dy) -> Optional[Tuple["_ScatterItem", int]]:
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        for item in c.scatters:
            if not item.visible:
                continue
            for i, (xi, yi) in enumerate(zip(item.xs, item.ys)):
                pt = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
                if math.hypot(pt.x() - pos.x(), pt.y() - pos.y()) <= item.size:
                    return (item, i)
        return None

    def _tangent_slope(self, item, xi):
        if not isinstance(item, (_LineItem, _FitItem)) or len(item.xs) < 2:
            return None
        xs = item.xs
        ys = item.ys
        n = len(xs)
        pos = bisect_left(xs, xi)
        if pos >= n:
            pos = n - 1
        elif pos > 0 and abs(xs[pos - 1] - xi) < abs(xs[pos] - xi):
            pos -= 1
        if pos == 0:
            ddx = xs[1] - xs[0]
            ddy = ys[1] - ys[0]
        elif pos == n - 1:
            ddx = xs[-1] - xs[-2]
            ddy = ys[-1] - ys[-2]
        else:
            ddx = xs[pos + 1] - xs[pos - 1]
            ddy = ys[pos + 1] - ys[pos - 1]
        return ddy / ddx if abs(ddx) > 1e-15 else None

    def _view_params(self, pr):
        c = self._chart
        x0 = to_log(c.vx0) if c.log_x else c.vx0
        x1 = to_log(c.vx1) if c.log_x else c.vx1
        y0 = to_log(c.vy0) if c.log_y else c.vy0
        y1 = to_log(c.vy1) if c.log_y else c.vy1
        dx = x1 - x0 if abs(x1 - x0) > 1e-300 else 1.0
        dy = y1 - y0 if abs(y1 - y0) > 1e-300 else 1.0
        return x0, x1, y0, y1, dx, dy

    def contextMenuEvent(self, ev):
        c = self._chart
        menu = QMenu(self)
        act_autofit = QAction(tr("chart_widget.btn_autofit_toggle"), menu)
        act_autofit.setCheckable(True)
        act_autofit.setChecked(c._autofit_enabled)
        act_autofit.triggered.connect(c.setAutofitEnabled)
        menu.addAction(act_autofit)
        act_latest = QAction(tr("chart_widget.btn_latest"), menu)
        act_latest.setCheckable(True)
        act_latest.setChecked(self._show_latest)
        act_latest.triggered.connect(self._set_latest)
        menu.addAction(act_latest)
        act_analytics = QAction(tr("chart_widget.btn_analytics"), menu)
        act_analytics.setCheckable(True)
        act_analytics.setChecked(self._show_analytics)
        act_analytics.triggered.connect(self._set_analytics)
        menu.addAction(act_analytics)
        act_legend = QAction(tr("chart_widget.ctx_legend"), menu)
        act_legend.setCheckable(True)
        act_legend.setChecked(c.show_legend)
        act_legend.triggered.connect(c.setLegendVisible)
        menu.addAction(act_legend)
        menu.addSeparator()
        log_menu = menu.addMenu(tr("chart_widget.ctx_log_scale"))
        act_log_x = QAction("Log X", log_menu)
        act_log_x.setCheckable(True)
        act_log_x.setChecked(c.log_x)
        act_log_x.triggered.connect(lambda v: c.setLogScale(x=v))
        log_menu.addAction(act_log_x)
        act_log_y = QAction("Log Y", log_menu)
        act_log_y.setCheckable(True)
        act_log_y.setChecked(c.log_y)
        act_log_y.triggered.connect(lambda v: c.setLogScale(y=v))
        log_menu.addAction(act_log_y)
        lock_menu = menu.addMenu(tr("chart_widget.ctx_zoom_lock"))
        grp_lock = QActionGroup(lock_menu)
        grp_lock.setExclusive(True)
        for key, lbl in (("both", tr("chart_widget.ctx_zoom_both")),
                          ("x", "X only"), ("y", "Y only")):
            act = QAction(lbl, lock_menu)
            act.setCheckable(True)
            act.setChecked(self._zoom_lock == key)
            act.triggered.connect(lambda checked, k=key: self.setZoomLock(k))
            lock_menu.addAction(act)
            grp_lock.addAction(act)
        menu.addSeparator()
        fit_menu = menu.addMenu(tr("chart_widget.ctx_approximation"))
        grp = QActionGroup(fit_menu)
        grp.setExclusive(True)
        for mode in get_fit_modes():
            act = QAction(mode.label, fit_menu)
            act.setCheckable(True)
            act.setChecked(c._active_fit_key == mode.key)
            act.triggered.connect(lambda checked, k=mode.key: c._on_fit_mode_selected(k))
            fit_menu.addAction(act)
            grp.addAction(act)
        menu.addSeparator()
        grid_menu = menu.addMenu(tr("chart_widget.ctx_grid"))
        x_menu = grid_menu.addMenu(tr("chart_widget.ctx_grid_x"))
        grp_x = QActionGroup(x_menu)
        grp_x.setExclusive(True)
        for key, px in _GRID_PRESETS_X:
            act = QAction(tr(key), x_menu)
            act.setCheckable(True)
            act.setChecked(c.grid_px_x == px)
            act.triggered.connect(lambda checked, v=px: c.setGridDensity(v, c.grid_px_y))
            x_menu.addAction(act)
            grp_x.addAction(act)
        y_menu = grid_menu.addMenu(tr("chart_widget.ctx_grid_y"))
        grp_y = QActionGroup(y_menu)
        grp_y.setExclusive(True)
        for key, px in _GRID_PRESETS_Y:
            act = QAction(tr(key), y_menu)
            act.setCheckable(True)
            act.setChecked(c.grid_px_y == px)
            act.triggered.connect(lambda checked, v=px: c.setGridDensity(c.grid_px_x, v))
            y_menu.addAction(act)
            grp_y.addAction(act)
        menu.addSeparator()
        if self._range_sel_x is not None:
            menu.addAction(tr("chart_widget.ctx_clear_range"), self.clearRangeSelection)
        menu.addAction(tr("chart_widget.ctx_export_csv"), c.exportCsv)
        menu.addAction(tr("chart_widget.ctx_export_img"), c.exportImage)
        menu.addSeparator()
        menu.addAction(tr("chart_widget.ctx_reset_view"), c.autofit)
        menu.exec_(ev.globalPos())

    def wheelEvent(self, ev: QWheelEvent):
        pr = self._plot_rect()
        if not pr.contains(ev.pos()):
            return
        c = self._chart
        x0, x1, y0, y1, dx, dy = self._view_params(pr)
        cx_log = x0 + (ev.pos().x() - pr.left()) / pr.width() * dx
        cy_log = y0 + (pr.bottom() - ev.pos().y()) / pr.height() * dy
        factor = 1.0 / _ZOOM_FACTOR if ev.angleDelta().y() > 0 else _ZOOM_FACTOR
        if self._zoom_lock in ("both", "x"):
            nvx0_log = cx_log + (x0 - cx_log) * factor
            nvx1_log = cx_log + (x1 - cx_log) * factor
            sx = abs(nvx1_log - nvx0_log)
            if math.isfinite(sx) and _ZOOM_MIN_SPAN <= sx <= _ZOOM_MAX_SPAN:
                c._vx0 = 10 ** nvx0_log if c.log_x else nvx0_log
                c._vx1 = 10 ** nvx1_log if c.log_x else nvx1_log
        if self._zoom_lock in ("both", "y"):
            nvy0_log = cy_log + (y0 - cy_log) * factor
            nvy1_log = cy_log + (y1 - cy_log) * factor
            sy = abs(nvy1_log - nvy0_log)
            if math.isfinite(sy) and _ZOOM_MIN_SPAN <= sy <= _ZOOM_MAX_SPAN:
                c._vy0 = 10 ** nvy0_log if c.log_y else nvy0_log
                c._vy1 = 10 ** nvy1_log if c.log_y else nvy1_log
        self.update()
        ev.accept()

    def mousePressEvent(self, ev: QMouseEvent):
        pr = self._plot_rect()
        c = self._chart
        x0, x1, y0, y1, dx, dy = self._view_params(pr)
        if ev.button() == Qt.MouseButton.LeftButton:
            hit_infline = self._find_infline_at(QPointF(ev.pos()), pr, x0, dx, y0, dy)
            if hit_infline:
                self._dragging_infline = hit_infline
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                ev.accept()
                return
            hit_scatter = self._find_scatter_point_at(QPointF(ev.pos()), pr, x0, dx, y0, dy)
            if hit_scatter:
                item, idx = hit_scatter
                item.selectPoint(idx)
                item.point_clicked.emit(item.xs[idx], item.ys[idx], idx)
                ev.accept()
                return
            if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._rb_start = QPointF(ev.pos())
                self._rb_end = QPointF(ev.pos())
                self._rb_active = True
                ev.accept()
                return
            self._pan_start = QPointF(ev.pos())
            self._pan_vx0 = c.vx0
            self._pan_vy0 = c.vy0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
        elif ev.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = QPointF(ev.pos())
            self._pan_vx0 = c.vx0
            self._pan_vy0 = c.vy0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()

    def mouseMoveEvent(self, ev: QMouseEvent):
        pr = self._plot_rect()
        self._mouse_pos = QPointF(ev.pos()) if pr.contains(ev.pos()) else None
        c = self._chart
        if self._dragging_infline is not None:
            x0, x1, y0, y1, dx, dy = self._view_params(pr)
            ln = self._dragging_infline
            if ln.horizontal:
                ly = y0 + (pr.bottom() - ev.pos().y()) / pr.height() * dy
                ln.value = 10 ** ly if c.log_y else ly
            else:
                lx = x0 + (ev.pos().x() - pr.left()) / pr.width() * dx
                ln.value = 10 ** lx if c.log_x else lx
            self._nearest_cache_key = None
            self.update()
            return
        if self._rb_active and self._rb_start is not None:
            self._rb_end = QPointF(ev.pos())
            self.update()
            return
        if self._pan_start is not None:
            x0, x1, y0, y1, dx, dy = self._view_params(pr)
            ddx = (ev.pos().x() - self._pan_start.x()) / pr.width() * dx
            ddy = (ev.pos().y() - self._pan_start.y()) / pr.height() * dy
            new_vx0_log = (to_log(self._pan_vx0) if c.log_x else self._pan_vx0) - ddx
            new_vy0_log = (to_log(self._pan_vy0) if c.log_y else self._pan_vy0) + ddy
            if c.log_x:
                c._vx0 = 10 ** new_vx0_log
                c._vx1 = 10 ** (new_vx0_log + dx)
            else:
                c._vx0 = new_vx0_log
                c._vx1 = new_vx0_log + dx
            if c.log_y:
                c._vy0 = 10 ** new_vy0_log
                c._vy1 = 10 ** (new_vy0_log + dy)
            else:
                c._vy0 = new_vy0_log
                c._vy1 = new_vy0_log + dy
            self._nearest_cache_key = None
            self.update()
            return
        if not (self._pan_start or self._rb_active or self._dragging_infline):
            if pr.contains(ev.pos()):
                x0, x1, y0, y1, dx, dy = self._view_params(pr)
                ln = self._find_infline_at(QPointF(ev.pos()), pr, x0, dx, y0, dy)
                if ln:
                    self.setCursor(Qt.CursorShape.SizeAllCursor if ln.horizontal else Qt.CursorShape.SizeHorCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if self._dragging_infline is not None:
            self._dragging_infline = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        if self._rb_active and self._rb_start is not None and self._rb_end is not None:
            pr = self._plot_rect()
            rb = QRectF(self._rb_start, self._rb_end).normalized()
            if rb.width() >= _RUBBERBAND_MIN_PX and rb.height() >= _RUBBERBAND_MIN_PX:
                c = self._chart
                x0, x1, y0, y1, dx, dy = self._view_params(pr)
                lx0 = x0 + (rb.left() - pr.left()) / pr.width() * dx
                lx1 = x0 + (rb.right() - pr.left()) / pr.width() * dx
                ly0 = y0 + (pr.bottom() - rb.bottom()) / pr.height() * dy
                ly1 = y0 + (pr.bottom() - rb.top()) / pr.height() * dy
                if c.log_x:
                    c._vx0, c._vx1 = 10 ** lx0, 10 ** lx1
                else:
                    c._vx0, c._vx1 = lx0, lx1
                if c.log_y:
                    c._vy0, c._vy1 = 10 ** ly0, 10 ** ly1
                else:
                    c._vy0, c._vy1 = ly0, ly1
            self._rb_active = False
            self._rb_start = None
            self._rb_end = None
            self.update()
            ev.accept()
            return
        self._pan_start = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        pr = self._plot_rect()
        if ev.button() == Qt.MouseButton.LeftButton and pr.contains(ev.pos()):
            c = self._chart
            x0, x1, y0, y1, dx, dy = self._view_params(pr)
            xv, _ = self._screen_to_data(ev.pos().x(), ev.pos().y(),
                                          x0, dx, y0, dy, pr, c.log_x, c.log_y)
            if self._range_sel_x is None:
                self._range_sel_x = (xv, xv)
            else:
                lo, hi = self._range_sel_x
                if abs(xv - lo) < abs(xv - hi):
                    self._range_sel_x = (xv, hi)
                else:
                    self._range_sel_x = (lo, xv)
            self.update()
        else:
            self._chart.autofit()

    def leaveEvent(self, ev):
        self._mouse_pos = None
        self.update()

    def _make_step_pts(self, xs, ys, x0, dx, y0, dy, pr, log_x, log_y):
        pts = []
        for i in range(len(xs)):
            pts.append(self._to_pt_log(xs[i], ys[i], x0, dx, y0, dy, pr, log_x, log_y))
            if i < len(xs) - 1:
                pts.append(self._to_pt_log(xs[i + 1], ys[i], x0, dx, y0, dy, pr, log_x, log_y))
        return pts

    def paintEvent(self, _):
        c = self._chart
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        bg = pal.window().color()
        fg = pal.windowText().color()
        ax_col = QColor(fg); ax_col.setAlpha(80)
        gr_col = QColor(fg); gr_col.setAlpha(80)
        lb_col = QColor(fg); lb_col.setAlpha(255)
        p.fillRect(self.rect(), bg)
        pr = self._plot_rect()
        x0, x1, y0, y1, dx, dy = self._view_params(pr)
        log_x, log_y = c.log_x, c.log_y
        has_r2 = self._has_right_axis()
        if not math.isfinite(dx) or abs(dx) < 1e-300:
            dx = 1.0
        if not math.isfinite(dy) or abs(dy) < 1e-300:
            dy = 1.0
        fm = QFontMetrics(c.font)
        p.setFont(c.font)
        n_x = max(2, pr.width() // max(1, c.grid_px_x))
        n_y = max(2, pr.height() // max(1, c.grid_px_y))
        if log_x:
            xt = nice_log_ticks(10 ** x0, 10 ** x1)
            xt_screen = [to_log(v) for v in xt]
        else:
            xt = nice_ticks(x0, x1, n_x)
            xt_screen = xt
        if log_y:
            yt = nice_log_ticks(10 ** y0, 10 ** y1)
            yt_screen = [to_log(v) for v in yt]
        else:
            yt = nice_ticks(y0, y1, n_y)
            yt_screen = yt
        p.setPen(QPen(gr_col, 1, Qt.PenStyle.DotLine))
        for tv in xt_screen:
            sx = int(pr.left() + (tv - x0) / dx * pr.width())
            p.drawLine(sx, pr.top(), sx, pr.bottom())
        for tv in yt_screen:
            sy = int(pr.bottom() - (tv - y0) / dy * pr.height())
            p.drawLine(pr.left(), sy, pr.right(), sy)
        p.setPen(QPen(ax_col, 1))
        p.drawRect(pr)
        p.setPen(lb_col)
        for tv, ts in zip(xt, xt_screen):
            sx = int(pr.left() + (ts - x0) / dx * pr.width())
            lbl = fmt(tv)
            lw = fm.horizontalAdvance(lbl)
            p.drawText(sx - lw // 2, pr.bottom() + fm.height() + 2, lbl)
        for tv, ts in zip(yt, yt_screen):
            sy = int(pr.bottom() - (ts - y0) / dy * pr.height())
            lbl = fmt(tv)
            lw = fm.horizontalAdvance(lbl)
            p.drawText(pr.left() - lw - 6, sy + fm.ascent() // 2, lbl)
        if has_r2:
            y0r = to_log(c.vy0_r) if c.log_y else c.vy0_r
            y1r = to_log(c.vy1_r) if c.log_y else c.vy1_r
            dyr = y1r - y0r if abs(y1r - y0r) > 1e-300 else 1.0
            n_yr = max(2, pr.height() // max(1, c.grid_px_y))
            if log_y:
                ytr = nice_log_ticks(10 ** y0r, 10 ** y1r)
                ytr_screen = [to_log(v) for v in ytr]
            else:
                ytr = nice_ticks(y0r, y1r, n_yr)
                ytr_screen = ytr
            r2_col = QColor(fg); r2_col.setAlpha(160)
            p.setPen(r2_col)
            for tv, ts in zip(ytr, ytr_screen):
                sy = int(pr.bottom() - (ts - y0r) / dyr * pr.height())
                lbl = fmt(tv)
                p.drawText(pr.right() + 6, sy + fm.ascent() // 2, lbl)
            if c.label_right:
                p.save()
                p.translate(self.width() - 8, pr.top() + pr.height() // 2)
                p.rotate(90)
                lw = fm.horizontalAdvance(c.label_right)
                p.drawText(-lw // 2, fm.ascent() // 2, c.label_right)
                p.restore()
        if c.label_bottom:
            lw = fm.horizontalAdvance(c.label_bottom)
            p.drawText(pr.left() + (pr.width() - lw) // 2, self.height() - 3, c.label_bottom)
        if c.label_left:
            p.save()
            p.translate(11, pr.top() + pr.height() // 2)
            p.rotate(-90)
            lw = fm.horizontalAdvance(c.label_left)
            p.drawText(-lw // 2, fm.ascent() // 2, c.label_left)
            p.restore()
        p.setClipRect(pr)
        if self._range_sel_x is not None:
            rx_lo, rx_hi = sorted(self._range_sel_x)
            lx_lo = to_log(rx_lo) if log_x else rx_lo
            lx_hi = to_log(rx_hi) if log_x else rx_hi
            sx_lo = max(pr.left(), int(pr.left() + (lx_lo - x0) / dx * pr.width()))
            sx_hi = min(pr.right(), int(pr.left() + (lx_hi - x0) / dx * pr.width()))
            if sx_hi > sx_lo:
                rsel_col = QColor(fg); rsel_col.setAlpha(_RANGE_SEL_ALPHA)
                p.fillRect(QRect(sx_lo, pr.top(), sx_hi - sx_lo, pr.height()), rsel_col)
                sel_border = QColor(fg); sel_border.setAlpha(100)
                p.setPen(QPen(sel_border, 1, Qt.PenStyle.DashLine))
                p.drawLine(sx_lo, pr.top(), sx_lo, pr.bottom())
                p.drawLine(sx_hi, pr.top(), sx_hi, pr.bottom())
        for ln in c.inflines:
            if not ln.visible:
                continue
            p.setPen(ln.pen)
            if ln.horizontal:
                ly = to_log(ln.value) if log_y else ln.value
                sy = int(pr.bottom() - (ly - y0) / dy * pr.height())
                p.drawLine(pr.left(), sy, pr.right(), sy)
            else:
                lx = to_log(ln.value) if log_x else ln.value
                sx = int(pr.left() + (lx - x0) / dx * pr.width())
                p.drawLine(sx, pr.top(), sx, pr.bottom())
        x_lo = min(c.vx0, c.vx1)
        x_hi = max(c.vx0, c.vx1)
        all_line_items = list(c.lines) + list(c.lines_r2)
        for fit in c.fits:
            if not fit.visible:
                continue
            fit._recompute(x_lo, x_hi, threaded=c._threaded_fit)
            if len(fit.xs) < 2:
                continue
            if _CY:
                flat, out_n = _cy_dec_screen(
                    fit.xs, fit.ys, len(fit.xs),
                    x0, dx, y0, dy,
                    float(pr.left()), float(pr.bottom()), float(pr.width()), float(pr.height()),
                    log_x, log_y,
                )
                path = QPainterPath()
                path.moveTo(flat[0], flat[1])
                for i in range(1, out_n):
                    path.lineTo(flat[i * 2], flat[i * 2 + 1])
                mid_flat_x = flat[(out_n // 2) * 2]
                mid_flat_y = flat[(out_n // 2) * 2 + 1]
            else:
                pts = [self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
                       for xi, yi in zip(fit.xs, fit.ys)]
                path = QPainterPath()
                path.moveTo(pts[0])
                for pt in pts[1:]:
                    path.lineTo(pt)
                mid_pt = pts[len(pts) // 2]
                mid_flat_x, mid_flat_y = mid_pt.x(), mid_pt.y()
            p.setPen(fit.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            if fit.show_formula:
                formula = fit.getFormula()
                if formula:
                    self._draw_formula_tag(p, QPointF(mid_flat_x, mid_flat_y), formula, fg, bg, fm)
        for item in c.lines:
            if not item.visible or not item.raw_visible or len(item.xs) < 2:
                continue
            if _CY and not item.step_mode:
                flat, out_n = _cy_dec_screen(
                    item.xs, item.ys, _DECIMATE_THRESHOLD,
                    x0, dx, y0, dy,
                    float(pr.left()), float(pr.bottom()), float(pr.width()), float(pr.height()),
                    log_x, log_y,
                )
                path = QPainterPath()
                path.moveTo(flat[0], flat[1])
                for i in range(1, out_n):
                    path.lineTo(flat[i * 2], flat[i * 2 + 1])
                if item.fill_under:
                    fill_path = QPainterPath(path)
                    baseline_y = pr.bottom() - (0 - y0) / dy * pr.height()
                    baseline_y = max(float(pr.top()), min(float(pr.bottom()), baseline_y))
                    fill_path.lineTo(flat[(out_n - 1) * 2], baseline_y)
                    fill_path.lineTo(flat[0], baseline_y)
                    fill_path.closeSubpath()
                    fill_col = QColor(item.pen.color())
                    fill_col.setAlpha(item.fill_alpha)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QBrush(fill_col))
                    p.drawPath(fill_path)
            else:
                dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
                if item.step_mode:
                    pts = self._make_step_pts(dxs, dys, x0, dx, y0, dy, pr, log_x, log_y)
                else:
                    pts = [self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
                           for xi, yi in zip(dxs, dys)]
                path = QPainterPath()
                path.moveTo(pts[0])
                for pt in pts[1:]:
                    path.lineTo(pt)
                if item.fill_under:
                    fill_path = QPainterPath(path)
                    baseline_y = pr.bottom() - (0 - y0) / dy * pr.height()
                    baseline_y = max(pr.top(), min(pr.bottom(), baseline_y))
                    fill_path.lineTo(pts[-1].x(), baseline_y)
                    fill_path.lineTo(pts[0].x(), baseline_y)
                    fill_path.closeSubpath()
                    fill_col = QColor(item.pen.color())
                    fill_col.setAlpha(item.fill_alpha)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QBrush(fill_col))
                    p.drawPath(fill_path)
            p.setPen(item.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        for item in c.lines_r2:
            if not item.visible or not item.raw_visible or len(item.xs) < 2:
                continue
            y0r = to_log(c.vy0_r) if log_y else c.vy0_r
            y1r = to_log(c.vy1_r) if log_y else c.vy1_r
            dyr = y1r - y0r if abs(y1r - y0r) > 1e-300 else 1.0
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            pts = [self._to_pt_log(xi, yi, x0, dx, y0r, dyr, pr, log_x, log_y)
                   for xi, yi in zip(dxs, dys)]
            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.setPen(item.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        all_scatters = list(c.scatters) + list(c.scatters_r2)
        for item in all_scatters:
            if not item.visible or not item.raw_visible or not item.xs:
                continue
            is_r2 = item in c.scatters_r2
            y0_eff = (to_log(c.vy0_r) if log_y else c.vy0_r) if is_r2 else y0
            dy_eff = ((to_log(c.vy1_r) - to_log(c.vy0_r)) if log_y else (c.vy1_r - c.vy0_r)) if is_r2 else dy
            if is_r2 and abs(dy_eff) < 1e-300:
                dy_eff = 1.0
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            if item.error_ys is not None:
                err_pen = QPen(item.color.darker(150), 1)
                p.setPen(err_pen)
                cap_px = 4
                for i, (xi, yi) in enumerate(zip(dxs, dys)):
                    if i >= len(item.error_ys):
                        break
                    ey = item.error_ys[i]
                    pt_top = self._to_pt_log(xi, yi + ey, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    pt_bot = self._to_pt_log(xi, yi - ey, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    p.drawLine(pt_bot, pt_top)
                    p.drawLine(int(pt_top.x()) - cap_px, int(pt_top.y()),
                               int(pt_top.x()) + cap_px, int(pt_top.y()))
                    p.drawLine(int(pt_bot.x()) - cap_px, int(pt_bot.y()),
                               int(pt_bot.x()) + cap_px, int(pt_bot.y()))
            if item.error_xs is not None:
                err_pen = QPen(item.color.darker(150), 1)
                p.setPen(err_pen)
                cap_px = 4
                for i, (xi, yi) in enumerate(zip(dxs, dys)):
                    if i >= len(item.error_xs):
                        break
                    ex = item.error_xs[i]
                    pt_l = self._to_pt_log(xi - ex, yi, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    pt_r = self._to_pt_log(xi + ex, yi, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    p.drawLine(pt_l, pt_r)
                    p.drawLine(int(pt_l.x()), int(pt_l.y()) - cap_px,
                               int(pt_l.x()), int(pt_l.y()) + cap_px)
                    p.drawLine(int(pt_r.x()), int(pt_r.y()) - cap_px,
                               int(pt_r.x()), int(pt_r.y()) + cap_px)
            r = item.size / 2.0
            for i, (xi, yi) in enumerate(zip(dxs, dys)):
                pt = self._to_pt_log(xi, yi, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                is_sel = (item.selected_idx is not None and i == item.selected_idx)
                if is_sel:
                    hi_col = QColor(item.color)
                    hi_col.setAlpha(80)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QBrush(hi_col))
                    p.drawEllipse(pt, r * 2.5, r * 2.5)
                p.setPen(QPen(item.color.darker(150), 2 if is_sel else 1))
                p.setBrush(QBrush(item.color))
                p.drawEllipse(pt, r, r)
            if item.annotations is not None:
                ann_col = QColor(fg); ann_col.setAlpha(200)
                p.setPen(ann_col)
                p.setFont(c.font)
                for i, (xi, yi) in enumerate(zip(item.xs, item.ys)):
                    if i >= len(item.annotations):
                        break
                    ann = item.annotations[i]
                    if not ann:
                        continue
                    pt = self._to_pt_log(xi, yi, x0, dx, y0_eff, dy_eff, pr, log_x, log_y)
                    p.drawText(int(pt.x()) + int(item.size / 2) + 2,
                               int(pt.y()) - 2, ann)
        if self._mouse_pos is not None and not self._rb_active:
            self._paint_crosshair(p, pr, x0, dx, y0, dy, fg, bg)
        if self._rb_active and self._rb_start and self._rb_end:
            rb = QRectF(self._rb_start, self._rb_end).normalized()
            rb_col = QColor(fg); rb_col.setAlpha(40)
            p.setBrush(QBrush(rb_col))
            rb_border = QColor(fg); rb_border.setAlpha(160)
            p.setPen(QPen(rb_border, 1, Qt.PenStyle.DashLine))
            p.drawRect(rb)
        p.setClipping(False)
        if self._show_latest:
            self._paint_latest_points(p, pr, x0, dx, y0, dy, fm)
        if self._show_analytics:
            self._paint_analytics(p, pr, fg, bg, fm)
        if c.show_legend:
            self._paint_legend(p, pr, fg, bg, fm)
        p.end()

    def _draw_formula_tag(self, p, pt, formula, fg, bg, fm):
        pad = 4
        tw = fm.horizontalAdvance(formula) + pad * 2
        th = fm.height() + pad
        tx = int(pt.x()) + 6
        ty = int(pt.y()) - th - 4
        bg_ = QColor(bg); bg_.setAlpha(200)
        br_ = QColor(fg); br_.setAlpha(80)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(br_, 1))
        p.drawRoundedRect(tx, ty, tw, th, 3, 3)
        p.setPen(fg)
        p.drawText(tx + pad, ty + fm.ascent() + pad // 2, formula)

    def _paint_latest_points(self, p, pr, x0, dx, y0, dy, fm):
        c = self._chart
        entries = []
        for it in c.lines:
            if it.visible and it.xs:
                entries.append((it.xs[-1], it.ys[-1], it.pen.color()))
        for it in c.scatters:
            if it.visible and it.xs:
                entries.append((it.xs[-1], it.ys[-1], it.color))
        p.setFont(c.font)
        pad = _LATEST_TAG_PAD
        rnd = _LATEST_TAG_ROUND
        th = fm.height()
        p.save()
        p.setClipRect(pr)
        log_x, log_y = c.log_x, c.log_y
        for xi, yi, color in entries:
            lx = to_log(xi) if log_x else xi
            ly = to_log(yi) if log_y else yi
            sx = pr.left() + (lx - x0) / dx * pr.width()
            sy = pr.bottom() - (ly - y0) / dy * pr.height()
            line_col = QColor(color); line_col.setAlpha(180)
            p.setPen(QPen(line_col, 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            if pr.top() <= sy <= pr.bottom():
                p.drawLine(int(sx), int(sy), int(sx), pr.bottom())
            if pr.left() <= sx <= pr.right():
                p.drawLine(pr.left(), int(sy), int(sx), int(sy))
        p.restore()
        for xi, yi, color in entries:
            lx = to_log(xi) if log_x else xi
            ly = to_log(yi) if log_y else yi
            sx = pr.left() + (lx - x0) / dx * pr.width()
            sy = pr.bottom() - (ly - y0) / dy * pr.height()
            lum = 0.299 * color.redF() + 0.587 * color.greenF() + 0.114 * color.blueF()
            txt_col = QColor("#000000") if lum > 0.5 else QColor("#ffffff")
            x_lbl = fmt(xi)
            xlw = fm.horizontalAdvance(x_lbl)
            x_tw = xlw + pad * 2
            xtx = int(sx) - x_tw // 2
            xty = pr.bottom() + 2
            if pr.left() - x_tw <= sx <= pr.right() + x_tw:
                p.setBrush(QBrush(color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(xtx, xty, x_tw, th, rnd, rnd)
                p.setPen(txt_col)
                p.drawText(xtx + pad, xty + fm.ascent(), x_lbl)
            y_lbl = fmt(yi)
            ylw = fm.horizontalAdvance(y_lbl)
            y_tw = ylw + pad * 2
            ytx = pr.left() - y_tw - 4
            yty = int(sy) - th // 2
            if pr.top() - th <= sy <= pr.bottom() + th:
                p.setBrush(QBrush(color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(ytx, yty, y_tw, th, rnd, rnd)
                p.setPen(txt_col)
                p.drawText(ytx + pad, yty + fm.ascent(), y_lbl)

    def _paint_crosshair(self, p, pr, x0, dx, y0, dy, fg, bg):
        mp = self._mouse_pos
        c = self._chart
        nearest = self._find_nearest(mp, pr, x0, dx, y0, dy)
        ch_col = QColor(fg); ch_col.setAlpha(80)
        p.setPen(QPen(ch_col, 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(int(mp.x()), pr.top(), int(mp.x()), pr.bottom())
        p.drawLine(pr.left(), int(mp.y()), pr.right(), int(mp.y()))
        if nearest is None:
            return
        xi, yi, _, item = nearest
        snap = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, c.log_x, c.log_y)
        dot_col = QColor(fg); dot_col.setAlpha(220)
        p.setPen(QPen(dot_col, 1))
        p.setBrush(QBrush(dot_col))
        p.drawEllipse(snap, _SNAP_DOT_R, _SNAP_DOT_R)
        slope = self._tangent_slope(item, xi)
        if slope is not None:
            half = (c.vx1 - c.vx0) * _TANGENT_HALF_FRAC
            tp0 = self._to_pt_log(xi - half, yi - slope * half, x0, dx, y0, dy, pr, c.log_x, c.log_y)
            tp1 = self._to_pt_log(xi + half, yi + slope * half, x0, dx, y0, dy, pr, c.log_x, c.log_y)
            tg_col = QColor(fg); tg_col.setAlpha(140)
            p.setPen(QPen(tg_col, 1, Qt.PenStyle.DotLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(tp0, tp1)
        self._paint_tooltip(p, pr, xi, yi, snap, fg, bg)

    def _paint_tooltip(self, p, pr, xi, yi, snap, fg, bg):
        fm = QFontMetrics(self._chart.font)
        p.setFont(self._chart.font)
        lx = f"{tr('chart_widget.tooltip_x')}: {fmt(xi)}"
        ly = f"{tr('chart_widget.tooltip_y')}: {fmt(yi)}"
        tw = max(fm.horizontalAdvance(lx), fm.horizontalAdvance(ly)) + 16
        th = fm.height() * 2 + 12
        tx = int(snap.x()) + _TOOLTIP_MARGIN
        ty = int(snap.y()) - th - 4
        if tx + tw > pr.right(): tx = int(snap.x()) - tw - _TOOLTIP_MARGIN
        if ty < pr.top(): ty = int(snap.y()) + 8
        bg_ = QColor(bg); bg_.setAlpha(215)
        br_ = QColor(fg); br_.setAlpha(100)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(br_, 1))
        p.drawRoundedRect(tx, ty, tw, th, 4, 4)
        p.setPen(fg)
        p.drawText(tx + 8, ty + fm.ascent() + 4, lx)
        p.drawText(tx + 8, ty + fm.ascent() + 4 + fm.height(), ly)

    def _paint_analytics(self, p, pr, fg, bg, fm):
        c = self._chart
        named = []
        for i, it in enumerate(c.lines):
            if it.xs and it.visible:
                named.append((it.label or tr("chart_widget.analytics_line", n=i + 1), it))
        for i, it in enumerate(c.scatters):
            if it.xs and it.visible:
                named.append((it.label or tr("chart_widget.analytics_scatter", n=i + 1), it))
        named = named[:_ANALYTICS_MAX_SERIES]
        if not named:
            return
        range_lo, range_hi = None, None
        if self._range_sel_x is not None:
            range_lo, range_hi = sorted(self._range_sel_x)
        row_keys = [
            "chart_widget.analytics_n", "chart_widget.analytics_xmin",
            "chart_widget.analytics_xmax", "chart_widget.analytics_ymin",
            "chart_widget.analytics_ymax", "chart_widget.analytics_mean",
            "chart_widget.analytics_std", "chart_widget.analytics_integral",
        ]
        row_lbls = [tr(k) for k in row_keys]
        table: List[List[str]] = []
        for _, it in named:
            if range_lo is not None and range_hi is not None:
                pairs = [(x, y) for x, y in zip(it.xs, it.ys) if range_lo <= x <= range_hi]
                fxs = [p[0] for p in pairs]
                fys = [p[1] for p in pairs]
            else:
                fxs, fys = it.xs, it.ys
            n = len(fxs)
            if n == 0:
                table.append(["-"] * len(row_keys))
                continue
            st = fmt(statistics.stdev(fys)) if n > 1 else "-"
            intg = fmt(trapezoid_integral(fxs, fys))
            table.append([
                str(n), fmt(min(fxs)), fmt(max(fxs)),
                fmt(min(fys)), fmt(max(fys)),
                fmt(statistics.mean(fys)), st, intg,
            ])
        lbl_w = max(fm.horizontalAdvance(l) for l in row_lbls) + 10
        val_w = max(fm.horizontalAdvance(v) for row in table for v in row) + 10
        hdr_h = fm.height() + 6
        rh = _ANALYTICS_ROW_H
        pad = _ANALYTICS_PAD
        n_ser = len(named)
        total_w = pad * 2 + lbl_w + val_w * n_ser
        total_h = pad * 2 + hdr_h + rh * len(row_lbls)
        ax = pr.right() - total_w - 4
        ay = pr.top() + 4
        bg_ = QColor(bg); bg_.setAlpha(210)
        brd_ = QColor(fg); brd_.setAlpha(70)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(brd_, 1))
        p.drawRoundedRect(ax, ay, total_w, total_h, 4, 4)
        bold_f = QFont(self._chart.font); bold_f.setBold(True)
        hdr_col = QColor(fg); hdr_col.setAlpha(220)
        lbl_col = QColor(fg); lbl_col.setAlpha(150)
        for ci, (name, _) in enumerate(named):
            cx = ax + pad + lbl_w + ci * val_w + val_w // 2
            p.setFont(bold_f)
            p.setPen(hdr_col)
            p.drawText(cx - fm.horizontalAdvance(name) // 2, ay + pad + fm.ascent(), name)
        p.setFont(self._chart.font)
        for ri, lbl in enumerate(row_lbls):
            ry = ay + pad + hdr_h + ri * rh
            p.setPen(lbl_col)
            p.drawText(ax + pad, ry + fm.ascent(), lbl)
            p.setPen(hdr_col)
            for ci, row in enumerate(table):
                val = row[ri]
                cx = ax + pad + lbl_w + ci * val_w
                vw = fm.horizontalAdvance(val)
                p.drawText(cx + (val_w - vw) // 2, ry + fm.ascent(), val)

    def _paint_legend(self, p, pr, fg, bg, fm):
        c = self._chart
        entries = []
        for i, it in enumerate(c.lines + c.lines_r2):
            if not it.visible:
                continue
            suffix = " (R)" if it in c.lines_r2 else ""
            entries.append((it.label or tr("chart_widget.legend_label", n=i + 1) + suffix,
                             it.pen.color(), False))
        for i, it in enumerate(c.scatters + c.scatters_r2):
            if not it.visible:
                continue
            suffix = " (R)" if it in c.scatters_r2 else ""
            entries.append((it.label or tr("chart_widget.legend_label",
                             n=len(c.lines) + len(c.lines_r2) + i + 1) + suffix,
                             it.color, True))
        for i, fit in enumerate(c.fits):
            if not fit.visible:
                continue
            entries.append((fit.label or tr("chart_widget.analytics_fit", n=i + 1),
                             fit.pen.color(), False))
        if not entries:
            return
        pad = _LEGEND_PAD
        sw = _LEGEND_SWATCH
        row_h = max(fm.height(), sw) + 4
        max_w = max(fm.horizontalAdvance(e[0]) for e in entries) + sw + pad * 2 + 6
        total_h = pad * 2 + row_h * len(entries)
        lx = pr.right() - max_w - 4
        ly = pr.bottom() - total_h - 4
        bg_ = QColor(bg); bg_.setAlpha(200)
        brd_ = QColor(fg); brd_.setAlpha(60)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(brd_, 1))
        p.drawRoundedRect(lx, ly, max_w, total_h, 4, 4)
        p.setFont(self._chart.font)
        for i, (label, color, is_scatter) in enumerate(entries):
            ry = ly + pad + i * row_h + (row_h - sw) // 2
            if is_scatter:
                p.setPen(QPen(color.darker(150), 1))
                p.setBrush(QBrush(color))
                p.drawEllipse(lx + pad, ry, sw, sw)
            else:
                p.setPen(QPen(color, 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                mid_y = ry + sw // 2
                p.drawLine(lx + pad, mid_y, lx + pad + sw, mid_y)
            txt_col = QColor(fg); txt_col.setAlpha(210)
            p.setPen(txt_col)
            p.drawText(lx + pad + sw + 6, ry + fm.ascent(), label)

    def grab_image(self) -> QPixmap:
        return self.grab()