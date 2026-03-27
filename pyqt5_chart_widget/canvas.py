from __future__ import annotations
import math
import statistics
from typing import List, Optional, Tuple, TYPE_CHECKING
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QRect, QPointF
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                          QFontMetrics, QPainterPath, QPixmap, QWheelEvent, QMouseEvent)
from .math_utils import nice_ticks, fmt
from .items import _LineItem, _ScatterItem, _FitItem, _InfLine
from .i18n import tr

if TYPE_CHECKING:
    from .chart_widget import ChartWidget

_ML, _MT, _MR, _MB = 58, 14, 20, 40
_ZOOM_FACTOR = 1.15
_SNAP_RADIUS_PX = 40
_TANGENT_HALF_FRAC = 0.18
_ANALYTICS_PAD = 8
_ANALYTICS_ROW_H = 17
_ANALYTICS_MAX_SERIES = 8
_TOOLTIP_MARGIN = 14
_SNAP_DOT_R = 5.0
_LEGEND_PAD = 8
_LEGEND_SWATCH = 12


class _PlotCanvas(QWidget):
    def __init__(self, chart: "ChartWidget"):
        super().__init__(chart)
        self._chart = chart
        self._pan_start: Optional[QPointF] = None
        self._pan_vx0 = 0.0
        self._pan_vy0 = 0.0
        self._mouse_pos: Optional[QPointF] = None
        self._show_analytics = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setMouseTracking(True)

    def toggleAnalytics(self):
        self._show_analytics = not self._show_analytics
        self.update()

    def _plot_rect(self) -> QRect:
        ml = _ML
        if self._chart.label_left:
            ml += 4
        return QRect(ml, _MT, max(1, self.width() - ml - _MR), max(1, self.height() - _MT - _MB))

    def _to_pt(self, xv, yv, x0, dx, y0, dy, pr) -> QPointF:
        return QPointF(
            pr.left() + (xv - x0) / dx * pr.width(),
            pr.bottom() - (yv - y0) / dy * pr.height(),
        )

    def _find_nearest(self, mouse, pr, x0, dx, y0, dy):
        best_d = float("inf")
        best = None
        c = self._chart
        all_items = [i for i in c.lines + c.scatters if i.visible]
        for item in all_items:
            for xi, yi in zip(item.xs, item.ys):
                pt = self._to_pt(xi, yi, x0, dx, y0, dy, pr)
                d = math.hypot(pt.x() - mouse.x(), pt.y() - mouse.y())
                if d < best_d:
                    best_d = d
                    best = (xi, yi, d, item)
        return best if best and best[2] <= _SNAP_RADIUS_PX else None

    def _tangent_slope(self, item, xi):
        if not isinstance(item, _LineItem) or len(item.xs) < 2:
            return None
        pts = list(zip(item.xs, item.ys))
        idx = min(range(len(pts)), key=lambda i: abs(pts[i][0] - xi))
        n = len(pts)
        if idx == 0:
            ddx, ddy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        elif idx == n - 1:
            ddx, ddy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
        else:
            ddx, ddy = pts[idx + 1][0] - pts[idx - 1][0], pts[idx + 1][1] - pts[idx - 1][1]
        return ddy / ddx if abs(ddx) > 1e-15 else None

    def wheelEvent(self, ev: QWheelEvent):
        pr = self._plot_rect()
        if not pr.contains(ev.pos()):
            return
        c = self._chart
        cx = c.vx0 + (ev.pos().x() - pr.left()) / pr.width() * (c.vx1 - c.vx0)
        cy = c.vy0 + (pr.bottom() - ev.pos().y()) / pr.height() * (c.vy1 - c.vy0)
        factor = 1.0 / _ZOOM_FACTOR if ev.angleDelta().y() > 0 else _ZOOM_FACTOR
        c._vx0 = cx + (c.vx0 - cx) * factor
        c._vx1 = cx + (c.vx1 - cx) * factor
        c._vy0 = cy + (c.vy0 - cy) * factor
        c._vy1 = cy + (c.vy1 - cy) * factor
        self.update()
        ev.accept()

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._pan_start = QPointF(ev.pos())
            self._pan_vx0 = self._chart.vx0
            self._pan_vy0 = self._chart.vy0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()

    def mouseMoveEvent(self, ev: QMouseEvent):
        pr = self._plot_rect()
        self._mouse_pos = QPointF(ev.pos()) if pr.contains(ev.pos()) else None
        if self._pan_start is not None:
            c = self._chart
            vdx = c.vx1 - c.vx0
            vdy = c.vy1 - c.vy0
            ddx = (ev.pos().x() - self._pan_start.x()) / pr.width() * vdx
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
        x0, x1 = c.vx0, c.vx1
        y0, y1 = c.vy0, c.vy1
        dx = x1 - x0 or 1.0
        dy = y1 - y0 or 1.0
        fm = QFontMetrics(c.font)
        p.setFont(c.font)
        xt = nice_ticks(x0, x1)
        yt = nice_ticks(y0, y1)
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
            sx = int(pr.left() + (tv - x0) / dx * pr.width())
            lbl = fmt(tv)
            lw = fm.horizontalAdvance(lbl)
            p.drawText(sx - lw // 2, pr.bottom() + fm.height() + 2, lbl)
        for tv in yt:
            sy = int(pr.bottom() - (tv - y0) / dy * pr.height())
            lbl = fmt(tv)
            lw = fm.horizontalAdvance(lbl)
            p.drawText(pr.left() - lw - 6, sy + fm.ascent() // 2, lbl)
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
        for ln in c.inflines:
            if not ln.visible:
                continue
            p.setPen(ln.pen)
            if ln.horizontal:
                sy = int(pr.bottom() - (ln.value - y0) / dy * pr.height())
                p.drawLine(pr.left(), sy, pr.right(), sy)
            else:
                sx = int(pr.left() + (ln.value - x0) / dx * pr.width())
                p.drawLine(sx, pr.top(), sx, pr.bottom())
        x_lo = min(x0, x1)
        x_hi = max(x0, x1)
        for fit in c.fits:
            if not fit.visible:
                continue
            fit._recompute(x_lo, x_hi)
            if len(fit.xs) < 2:
                continue
            pts = [self._to_pt(xi, yi, x0, dx, y0, dy, pr) for xi, yi in zip(fit.xs, fit.ys)]
            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.setPen(fit.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        for item in c.lines:
            if not item.visible or len(item.xs) < 2:
                continue
            pts = [self._to_pt(xi, yi, x0, dx, y0, dy, pr) for xi, yi in zip(item.xs, item.ys)]
            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.setPen(item.pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        for item in c.scatters:
            if not item.visible or not item.xs:
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
        if c.show_legend:
            self._paint_legend(p, pr, fg, bg, fm)
        p.end()

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
        snap = self._to_pt(xi, yi, x0, dx, y0, dy, pr)
        dot_col = QColor(fg); dot_col.setAlpha(220)
        p.setPen(QPen(dot_col, 1))
        p.setBrush(QBrush(dot_col))
        p.drawEllipse(snap, _SNAP_DOT_R, _SNAP_DOT_R)
        slope = self._tangent_slope(item, xi)
        if slope is not None:
            half = (c.vx1 - c.vx0) * _TANGENT_HALF_FRAC
            tp0 = self._to_pt(xi - half, yi - slope * half, x0, dx, y0, dy, pr)
            tp1 = self._to_pt(xi + half, yi + slope * half, x0, dx, y0, dy, pr)
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
                label = it.label or tr("chart_widget.analytics_line", n=i + 1)
                named.append((label, it))
        for i, it in enumerate(c.scatters):
            if it.xs and it.visible:
                label = it.label or tr("chart_widget.analytics_scatter", n=i + 1)
                named.append((label, it))
        named = named[:_ANALYTICS_MAX_SERIES]
        if not named:
            return
        row_keys = [
            "chart_widget.analytics_n", "chart_widget.analytics_xmin",
            "chart_widget.analytics_xmax", "chart_widget.analytics_ymin",
            "chart_widget.analytics_ymax", "chart_widget.analytics_mean",
            "chart_widget.analytics_std",
        ]
        row_lbls = [tr(k) for k in row_keys]
        table: List[List[str]] = []
        for _, it in named:
            n = len(it.xs)
            st = fmt(statistics.stdev(it.ys)) if n > 1 else "—"
            table.append([
                str(n), fmt(min(it.xs)), fmt(max(it.xs)),
                fmt(min(it.ys)), fmt(max(it.ys)),
                fmt(statistics.mean(it.ys)), st,
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
        for i, it in enumerate(c.lines):
            if not it.visible:
                continue
            label = it.label or tr("chart_widget.legend_label", n=i + 1)
            color = it.pen.color()
            entries.append((label, color, False))
        for i, it in enumerate(c.scatters):
            if not it.visible:
                continue
            label = it.label or tr("chart_widget.legend_label", n=len(c.lines) + i + 1)
            entries.append((label, it.color, True))
        for i, fit in enumerate(c.fits):
            if not fit.visible:
                continue
            label = fit.label or tr("chart_widget.analytics_fit", n=i + 1)
            entries.append((label, fit.pen.color(), False))
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
