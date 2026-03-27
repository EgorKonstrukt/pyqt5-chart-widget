from __future__ import annotations
import csv
import math
import statistics
from typing import List, Optional, Tuple, Union
from PyQt5.QtWidgets import (QWidget, QSizePolicy, QFileDialog,
                              QToolButton, QHBoxLayout, QVBoxLayout, QStyle)
from PyQt5.QtCore import Qt, QRect, QPointF, QSize
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                          QFontMetrics, QPainterPath, QWheelEvent,
                          QMouseEvent, QPixmap)

try:
    from i18n import tr
except ImportError:
    _FB: dict = {
        "chart_widget.btn_fit":           "Fit",
        "chart_widget.btn_fit_tip":       "Auto-fit view to data (double-click on chart)",
        "chart_widget.btn_csv":           "CSV",
        "chart_widget.btn_csv_tip":       "Export data to CSV",
        "chart_widget.btn_img":           "Image",
        "chart_widget.btn_img_tip":       "Export chart as PNG image",
        "chart_widget.btn_analytics":     "Stats",
        "chart_widget.btn_analytics_tip": "Show / hide analytics panel",
        "chart_widget.csv_title":         "Export to CSV",
        "chart_widget.csv_filter":        "CSV files (*.csv);;All files (*)",
        "chart_widget.img_title":         "Export image",
        "chart_widget.img_filter":        "PNG images (*.png);;All files (*)",
        "chart_widget.tooltip_x":         "X",
        "chart_widget.tooltip_y":         "Y",
        "chart_widget.analytics_line":    "Line {n}",
        "chart_widget.analytics_scatter": "Scatter {n}",
        "chart_widget.analytics_n":       "n",
        "chart_widget.analytics_xmin":    "x min",
        "chart_widget.analytics_xmax":    "x max",
        "chart_widget.analytics_ymin":    "y min",
        "chart_widget.analytics_ymax":    "y max",
        "chart_widget.analytics_mean":    "mean y",
        "chart_widget.analytics_std":     "std y",
    }
    def tr(key: str, **kwargs) -> str:
        text = _FB.get(key, key.split(".")[-1])
        return text.format(**kwargs) if kwargs else text

_ML, _MT, _MR, _MB   = 58, 14, 20, 40
_ZOOM_FACTOR          = 1.15
_SNAP_RADIUS_PX       = 40
_TANGENT_HALF_FRAC    = 0.18
_ANALYTICS_PAD        = 8
_ANALYTICS_ROW_H      = 17
_ANALYTICS_MAX_SERIES = 6
_TOOLTIP_MARGIN       = 14
_SNAP_DOT_R           = 5.0


def _nice_ticks(lo: float, hi: float, n: int = 7) -> List[float]:
    if hi <= lo:
        return [lo]
    span = hi - lo
    raw  = span / max(n - 1, 1)
    mag  = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = mag
    for s in (mag, mag * 2, mag * 2.5, mag * 5, mag * 10):
        if span / s <= n + 1:
            step = s
            break
    start = math.floor(lo / step) * step
    ticks: List[float] = []
    v = start
    while v <= hi + step * 0.001:
        if v >= lo - step * 0.001:
            ticks.append(round(v, 10))
        v = round(v + step, 10)
    return ticks


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1000 or (abs(v) < 0.001 and v != 0):
        return f"{v:.3g}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3g}"


class _InfLine:
    def __init__(self, chart: "ChartWidget", horizontal: bool, value: float, pen: QPen):
        self._chart     = chart
        self.horizontal = horizontal
        self.value      = value
        self.pen        = pen
        self.visible    = False
    def setValue(self, v: float):
        self.value = v
        self._chart.update()
    def setVisible(self, v: bool):
        self.visible = v
        self._chart.update()


class _ScatterItem:
    def __init__(self, chart: "ChartWidget", size: int, color: QColor):
        self._chart = chart
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.size  = size
        self.color = color
    def setData(self, x=None, y=None, **_):
        self.xs = list(x) if x is not None else []
        self.ys = list(y) if y is not None else []
        self._chart._schedule_autofit()


class _LineItem:
    def __init__(self, chart: "ChartWidget", pen: QPen):
        self._chart = chart
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.pen = pen
    def setData(self, xs=None, ys=None):
        self.xs = list(xs) if xs is not None else []
        self.ys = list(ys) if ys is not None else []
        self._chart._schedule_autofit()


_AnyItem = Union[_LineItem, _ScatterItem]


class _PlotCanvas(QWidget):
    def __init__(self, chart: "ChartWidget"):
        super().__init__(chart)
        self._chart          = chart
        self._pan_start:     Optional[QPointF] = None
        self._pan_vx0        = 0.0
        self._pan_vy0        = 0.0
        self._mouse_pos:     Optional[QPointF] = None
        self._show_analytics = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setMouseTracking(True)

    def toggleAnalytics(self):
        self._show_analytics = not self._show_analytics
        self.update()

    def _plot_rect(self) -> QRect:
        return QRect(_ML, _MT,
                     max(1, self.width()  - _ML - _MR),
                     max(1, self.height() - _MT - _MB))

    def _to_pt(self, xv: float, yv: float,
               x0: float, dx: float, y0: float, dy: float, pr: QRect) -> QPointF:
        return QPointF(
            pr.left()   + (xv - x0) / dx * pr.width(),
            pr.bottom() - (yv - y0) / dy * pr.height(),
        )

    def _find_nearest(self, mouse: QPointF, pr: QRect,
                      x0: float, dx: float, y0: float, dy: float
                      ) -> Optional[Tuple[float, float, float, _AnyItem]]:
        best_d = float("inf")
        best   = None
        c      = self._chart
        for item in c._lines + c._scatters:
            for xi, yi in zip(item.xs, item.ys):
                pt = self._to_pt(xi, yi, x0, dx, y0, dy, pr)
                d  = math.hypot(pt.x() - mouse.x(), pt.y() - mouse.y())
                if d < best_d:
                    best_d = d
                    best   = (xi, yi, d, item)
        return best if best and best[2] <= _SNAP_RADIUS_PX else None

    def _tangent_slope(self, item: _AnyItem, xi: float) -> Optional[float]:
        if not isinstance(item, _LineItem) or len(item.xs) < 2:
            return None
        pts = list(zip(item.xs, item.ys))
        idx = min(range(len(pts)), key=lambda i: abs(pts[i][0] - xi))
        n   = len(pts)
        if idx == 0:
            ddx, ddy = pts[1][0] - pts[0][0],   pts[1][1] - pts[0][1]
        elif idx == n - 1:
            ddx, ddy = pts[-1][0] - pts[-2][0],  pts[-1][1] - pts[-2][1]
        else:
            ddx, ddy = pts[idx+1][0] - pts[idx-1][0], pts[idx+1][1] - pts[idx-1][1]
        return ddy / ddx if abs(ddx) > 1e-15 else None

    def wheelEvent(self, ev: QWheelEvent):
        pr = self._plot_rect()
        if not pr.contains(ev.pos()):
            return
        c      = self._chart
        cx     = c._vx0 + (ev.pos().x() - pr.left()) / pr.width()  * (c._vx1 - c._vx0)
        cy     = c._vy0 + (pr.bottom() - ev.pos().y()) / pr.height() * (c._vy1 - c._vy0)
        factor = 1.0 / _ZOOM_FACTOR if ev.angleDelta().y() > 0 else _ZOOM_FACTOR
        c._vx0 = cx + (c._vx0 - cx) * factor
        c._vx1 = cx + (c._vx1 - cx) * factor
        c._vy0 = cy + (c._vy0 - cy) * factor
        c._vy1 = cy + (c._vy1 - cy) * factor
        self.update()
        ev.accept()

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._pan_start = QPointF(ev.pos())
            self._pan_vx0   = self._chart._vx0
            self._pan_vy0   = self._chart._vy0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()

    def mouseMoveEvent(self, ev: QMouseEvent):
        pr = self._plot_rect()
        self._mouse_pos = QPointF(ev.pos()) if pr.contains(ev.pos()) else None
        if self._pan_start is not None:
            c   = self._chart
            vdx = c._vx1 - c._vx0
            vdy = c._vy1 - c._vy0
            ddx = (ev.pos().x() - self._pan_start.x()) / pr.width()  * vdx
            ddy = (ev.pos().y() - self._pan_start.y()) / pr.height() * vdy
            c._vx0 = self._pan_vx0 - ddx
            c._vx1 = self._pan_vx0 - ddx + vdx
            c._vy0 = self._pan_vy0 + ddy
            c._vy1 = self._pan_vy0 + ddy + vdy
        self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._pan_start = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        self._chart.autofit()

    def leaveEvent(self, ev):
        self._mouse_pos = None
        self.update()

    def paintEvent(self, _):
        c      = self._chart
        p      = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal    = self.palette()
        bg     = pal.window().color()
        fg     = pal.windowText().color()
        ax_col = QColor(fg); ax_col.setAlpha(80)
        gr_col = QColor(fg); gr_col.setAlpha(40)
        lb_col = QColor(fg); lb_col.setAlpha(200)
        p.fillRect(self.rect(), bg)
        pr     = self._plot_rect()
        x0, x1 = c._vx0, c._vx1
        y0, y1 = c._vy0, c._vy1
        dx     = x1 - x0 or 1.0
        dy     = y1 - y0 or 1.0
        fm     = QFontMetrics(c._font)
        p.setFont(c._font)
        xt     = _nice_ticks(x0, x1)
        yt     = _nice_ticks(y0, y1)
        p.setPen(QPen(gr_col, 1, Qt.PenStyle.DotLine))
        for tv in xt:
            sx = int(pr.left() + (tv - x0) / dx * pr.width())
            p.drawLine(sx, pr.top(), sx, pr.bottom())
        for tv in yt:
            sy = int(pr.bottom() - (tv - y0) / dy * pr.height())
            p.drawLine(pr.left(), sy, pr.right(), sy)
        p.setPen(QPen(ax_col, 1))
        p.drawRect(pr)
        p.setPen(lb_col)
        for tv in xt:
            sx  = int(pr.left() + (tv - x0) / dx * pr.width())
            lbl = _fmt(tv)
            lw  = fm.horizontalAdvance(lbl)
            p.drawText(sx - lw // 2, pr.bottom() + fm.height() + 2, lbl)
        for tv in yt:
            sy  = int(pr.bottom() - (tv - y0) / dy * pr.height())
            lbl = _fmt(tv)
            lw  = fm.horizontalAdvance(lbl)
            p.drawText(pr.left() - lw - 6, sy + fm.ascent() // 2, lbl)
        if c._label_bottom:
            lw = fm.horizontalAdvance(c._label_bottom)
            p.drawText(pr.left() + (pr.width() - lw) // 2, self.height() - 3, c._label_bottom)
        if c._label_left:
            p.save()
            p.translate(11, pr.top() + pr.height() // 2)
            p.rotate(-90)
            lw = fm.horizontalAdvance(c._label_left)
            p.drawText(-lw // 2, fm.ascent() // 2, c._label_left)
            p.restore()
        p.setClipRect(pr)
        for ln in c._inflines:
            if not ln.visible:
                continue
            p.setPen(ln.pen)
            if ln.horizontal:
                sy = int(pr.bottom() - (ln.value - y0) / dy * pr.height())
                p.drawLine(pr.left(), sy, pr.right(), sy)
            else:
                sx = int(pr.left() + (ln.value - x0) / dx * pr.width())
                p.drawLine(sx, pr.top(), sx, pr.bottom())
        for item in c._lines:
            if len(item.xs) < 2:
                continue
            pts  = [self._to_pt(xi, yi, x0, dx, y0, dy, pr)
                    for xi, yi in zip(item.xs, item.ys)]
            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.setPen(item.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        for item in c._scatters:
            if not item.xs:
                continue
            p.setPen(QPen(item.color.darker(150), 1))
            p.setBrush(QBrush(item.color))
            r = item.size / 2.0
            for xi, yi in zip(item.xs, item.ys):
                pt = self._to_pt(xi, yi, x0, dx, y0, dy, pr)
                p.drawEllipse(pt, r, r)
        if self._mouse_pos is not None:
            self._paint_crosshair(p, pr, x0, dx, y0, dy, fg, bg)
        p.setClipping(False)
        if self._show_analytics:
            self._paint_analytics(p, pr, fg, bg, fm)
        p.end()

    def _paint_crosshair(self, p: QPainter, pr: QRect,
                         x0: float, dx: float, y0: float, dy: float,
                         fg: QColor, bg: QColor):
        mp      = self._mouse_pos
        c       = self._chart
        nearest = self._find_nearest(mp, pr, x0, dx, y0, dy)
        ch_col  = QColor(fg); ch_col.setAlpha(80)
        p.setPen(QPen(ch_col, 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(int(mp.x()), pr.top(), int(mp.x()), pr.bottom())
        p.drawLine(pr.left(), int(mp.y()), pr.right(), int(mp.y()))
        if nearest is None:
            return
        xi, yi, _, item = nearest
        snap    = self._to_pt(xi, yi, x0, dx, y0, dy, pr)
        dot_col = QColor(fg); dot_col.setAlpha(220)
        p.setPen(QPen(dot_col, 1))
        p.setBrush(QBrush(dot_col))
        p.drawEllipse(snap, _SNAP_DOT_R, _SNAP_DOT_R)
        slope = self._tangent_slope(item, xi)
        if slope is not None:
            half = (c._vx1 - c._vx0) * _TANGENT_HALF_FRAC
            tp0  = self._to_pt(xi - half, yi - slope * half, x0, dx, y0, dy, pr)
            tp1  = self._to_pt(xi + half, yi + slope * half, x0, dx, y0, dy, pr)
            tg_col = QColor(fg); tg_col.setAlpha(140)
            p.setPen(QPen(tg_col, 1, Qt.PenStyle.DotLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(tp0, tp1)
        self._paint_tooltip(p, pr, xi, yi, snap, fg, bg)

    def _paint_tooltip(self, p: QPainter, pr: QRect,
                       xi: float, yi: float, snap: QPointF,
                       fg: QColor, bg: QColor):
        fm  = QFontMetrics(self._chart._font)
        p.setFont(self._chart._font)
        lx  = f"{tr('chart_widget.tooltip_x')}: {_fmt(xi)}"
        ly  = f"{tr('chart_widget.tooltip_y')}: {_fmt(yi)}"
        tw  = max(fm.horizontalAdvance(lx), fm.horizontalAdvance(ly)) + 16
        th  = fm.height() * 2 + 12
        tx  = int(snap.x()) + _TOOLTIP_MARGIN
        ty  = int(snap.y()) - th - 4
        if tx + tw > pr.right(): tx = int(snap.x()) - tw - _TOOLTIP_MARGIN
        if ty < pr.top():        ty = int(snap.y()) + 8
        bg_ = QColor(bg); bg_.setAlpha(215)
        br_ = QColor(fg); br_.setAlpha(100)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(br_, 1))
        p.drawRoundedRect(tx, ty, tw, th, 4, 4)
        p.setPen(fg)
        p.drawText(tx + 8, ty + fm.ascent() + 4, lx)
        p.drawText(tx + 8, ty + fm.ascent() + 4 + fm.height(), ly)

    def _paint_analytics(self, p: QPainter, pr: QRect,
                         fg: QColor, bg: QColor, fm: QFontMetrics):
        c = self._chart
        named: List[Tuple[str, _AnyItem]] = []
        for i, it in enumerate(c._lines):
            if it.xs:
                named.append((tr("chart_widget.analytics_line",    n=i + 1), it))
        for i, it in enumerate(c._scatters):
            if it.xs:
                named.append((tr("chart_widget.analytics_scatter", n=i + 1), it))
        named = named[:_ANALYTICS_MAX_SERIES]
        if not named:
            return
        row_keys = [
            "chart_widget.analytics_n",
            "chart_widget.analytics_xmin",
            "chart_widget.analytics_xmax",
            "chart_widget.analytics_ymin",
            "chart_widget.analytics_ymax",
            "chart_widget.analytics_mean",
            "chart_widget.analytics_std",
        ]
        row_lbls = [tr(k) for k in row_keys]
        table: List[List[str]] = []
        for _, it in named:
            n  = len(it.xs)
            st = _fmt(statistics.stdev(it.ys)) if n > 1 else "—"
            table.append([
                str(n),
                _fmt(min(it.xs)), _fmt(max(it.xs)),
                _fmt(min(it.ys)), _fmt(max(it.ys)),
                _fmt(statistics.mean(it.ys)), st,
            ])
        lbl_w   = max(fm.horizontalAdvance(l) for l in row_lbls) + 10
        val_w   = max(fm.horizontalAdvance(v) for row in table for v in row) + 10
        hdr_h   = fm.height() + 6
        rh      = _ANALYTICS_ROW_H
        pad     = _ANALYTICS_PAD
        n_ser   = len(named)
        total_w = pad * 2 + lbl_w + val_w * n_ser
        total_h = pad * 2 + hdr_h + rh * len(row_lbls)
        ax      = pr.right()  - total_w - 4
        ay      = pr.top()    + 4
        bg_     = QColor(bg); bg_.setAlpha(210)
        brd_    = QColor(fg); brd_.setAlpha(70)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(brd_, 1))
        p.drawRoundedRect(ax, ay, total_w, total_h, 4, 4)
        bold_f  = QFont(self._chart._font); bold_f.setBold(True)
        hdr_col = QColor(fg); hdr_col.setAlpha(220)
        lbl_col = QColor(fg); lbl_col.setAlpha(150)
        for ci, (name, _) in enumerate(named):
            cx = ax + pad + lbl_w + ci * val_w + val_w // 2
            p.setFont(bold_f)
            p.setPen(hdr_col)
            p.drawText(cx - fm.horizontalAdvance(name) // 2, ay + pad + fm.ascent(), name)
        p.setFont(self._chart._font)
        for ri, lbl in enumerate(row_lbls):
            ry = ay + pad + hdr_h + ri * rh
            p.setPen(lbl_col)
            p.drawText(ax + pad, ry + fm.ascent(), lbl)
            p.setPen(hdr_col)
            for ci, row in enumerate(table):
                val = row[ri]
                cx  = ax + pad + lbl_w + ci * val_w
                vw  = fm.horizontalAdvance(val)
                p.drawText(cx + (val_w - vw) // 2, ry + fm.ascent(), val)

    def grab_image(self) -> QPixmap:
        return self.grab()


class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines:    List[_LineItem]    = []
        self._scatters: List[_ScatterItem] = []
        self._inflines: List[_InfLine]     = []
        self._label_left   = ""
        self._label_bottom = ""
        self._font         = QFont("Arial", 8)
        self._vx0 = 0.0;  self._vx1 = 1.0
        self._vy0 = 0.0;  self._vy1 = 1.0
        self._canvas    = _PlotCanvas(self)
        self._btn_fit   = self._make_btn("SP_FileDialogContentsView",
                                         "chart_widget.btn_fit",
                                         "chart_widget.btn_fit_tip",
                                         self.autofit)
        self._btn_stats = self._make_btn("SP_FileDialogInfoView",
                                         "chart_widget.btn_analytics",
                                         "chart_widget.btn_analytics_tip",
                                         self._canvas.toggleAnalytics)
        self._btn_csv   = self._make_btn("SP_DialogSaveButton",
                                         "chart_widget.btn_csv",
                                         "chart_widget.btn_csv_tip",
                                         self.exportCsv)
        self._btn_img   = self._make_btn("SP_DialogSaveButton",
                                         "chart_widget.btn_img",
                                         "chart_widget.btn_img_tip",
                                         self.exportImage)
        tb = QHBoxLayout()
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(2)
        tb.addStretch()
        for btn in (self._btn_fit, self._btn_stats, self._btn_csv, self._btn_img):
            tb.addWidget(btn)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addLayout(tb)
        root.addWidget(self._canvas, 1)
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _make_btn(self, icon_sp: str, lbl_key: str,
                  tip_key: str, slot) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(tr(lbl_key))
        btn.setToolTip(tr(tip_key))
        btn.setIcon(self.style().standardIcon(
            getattr(QStyle.StandardPixmap, icon_sp)))
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setFixedHeight(24)
        btn.clicked.connect(slot)
        return btn

    def showGrid(self, x: bool = True, y: bool = True, alpha: float = 0.3):
        self._canvas.update()

    def setLabel(self, side: str, text: str):
        if side == "left":     self._label_left   = text
        elif side == "bottom": self._label_bottom = text
        self._canvas.update()

    def plot(self, color: str = "#e74c3c", width: int = 2) -> _LineItem:
        pen = QPen(QColor(color), width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        item = _LineItem(self, pen)
        self._lines.append(item)
        return item

    def addScatter(self, size: int = 10, color: str = "#27ae60") -> _ScatterItem:
        item = _ScatterItem(self, size, QColor(color))
        self._scatters.append(item)
        return item

    def addItem(self, item):
        if isinstance(item, _LineItem) and item not in self._lines:
            self._lines.append(item)
        elif isinstance(item, _ScatterItem) and item not in self._scatters:
            self._scatters.append(item)
        self._canvas.update()

    def addLine(self, y: Optional[float] = None, x: Optional[float] = None,
                color: str = "#f39c12", width: int = 1, dashed: bool = True) -> _InfLine:
        horiz = y is not None
        val   = y if horiz else (x if x is not None else 0.0)
        pen   = QPen(QColor(color), width,
                     Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        ln    = _InfLine(self, horiz, val, pen)
        self._inflines.append(ln)
        return ln

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

    def exportCsv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, tr("chart_widget.csv_title"), "chart_data.csv",
            tr("chart_widget.csv_filter"))
        if not path:
            return
        series = []
        for i, it in enumerate(self._lines):
            if it.xs:
                series.append((f"line{i}_x", f"line{i}_y", it.xs, it.ys))
        for i, it in enumerate(self._scatters):
            if it.xs:
                series.append((f"scatter{i}_x", f"scatter{i}_y", it.xs, it.ys))
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