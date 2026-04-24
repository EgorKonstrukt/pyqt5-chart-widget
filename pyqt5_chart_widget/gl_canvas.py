from __future__ import annotations
import math
import array
from typing import List, Optional, Tuple, TYPE_CHECKING
from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                         QFontMetrics, QPainterPath, QPixmap,
                         QWheelEvent, QMouseEvent, QPolygonF,
                         QOpenGLVersionProfile, QSurfaceFormat)
from PyQt5.QtOpenGL import QGLFormat
from OpenGL.GL import (
    glViewport, glClearColor, glClear, glEnable, glDisable,
    glBlendFunc, glLineWidth, glPointSize,
    glUseProgram, glUniformMatrix4fv, glUniform4f, glUniform1i,
    glUniform1f, glUniform2f,
    glGenBuffers, glBindBuffer, glBufferData, glBufferSubData,
    glVertexAttribPointer, glEnableVertexAttribArray, glDisableVertexAttribArray,
    glDrawArrays, glDrawElements,
    glCreateShader, glShaderSource, glCompileShader,
    glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram,
    glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glDeleteTextures, glActiveTexture,
    glScissor, glColorMask, glDepthMask, glStencilMask,
    glLineStipple, GL_LINE_STIPPLE,
    GL_COLOR_BUFFER_BIT, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LINES, GL_LINE_STRIP, GL_POINTS, GL_TRIANGLES, GL_TRIANGLE_FAN,
    GL_FLOAT, GL_FALSE, GL_TRUE,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    GL_COMPILE_STATUS, GL_LINK_STATUS,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW,
    GL_TEXTURE_2D, GL_TEXTURE0, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
    GL_SCISSOR_TEST, GL_MULTISAMPLE, GL_POINT_SMOOTH
)
import ctypes
import numpy as np
from .math_utils import (nice_ticks, nice_log_ticks, to_log, decimated,
                         fmt, get_fit_modes, trapezoid_integral)
from .items import _LineItem, _ScatterItem, _FitItem, _InfLine, _FunctionItem, _RulerItem
from .i18n import tr

if TYPE_CHECKING:
    from .chart_widget import ChartWidget

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
_SCREEN_Y_CLAMP = 32768.0
_GRID_PRESETS_X = [("chart_widget.ctx_sparse", 120), ("chart_widget.ctx_normal", 80), ("chart_widget.ctx_dense", 50)]
_GRID_PRESETS_Y = [("chart_widget.ctx_sparse", 100), ("chart_widget.ctx_normal", 60), ("chart_widget.ctx_dense", 40)]

_VERT_SRC = """
#version 120
attribute vec2 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
}
"""

_FRAG_SRC = """
#version 120
uniform vec4 uColor;
void main() {
    gl_FragColor = uColor;
}
"""

_VERT_CIRCLE_SRC = """
#version 120
attribute vec2 aPos;
attribute vec2 aCenter;
attribute float aRadius;
attribute vec4 aColor;
uniform mat4 uMVP;
varying vec2 vLocal;
varying vec4 vColor;
void main() {
    vLocal = aPos;
    vColor = aColor;
    gl_Position = uMVP * vec4(aCenter + aPos * aRadius, 0.0, 1.0);
}
"""

_FRAG_CIRCLE_SRC = """
#version 120
varying vec2 vLocal;
varying vec4 vColor;
void main() {
    float d = length(vLocal);
    if (d > 1.0) discard;
    float alpha = smoothstep(1.0, 0.8, d);
    gl_FragColor = vec4(vColor.rgb, vColor.a * alpha);
}
"""

_VERT_TEX_SRC = """
#version 120
attribute vec2 aPos;
attribute vec2 aUV;
uniform mat4 uMVP;
varying vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
}
"""

_FRAG_TEX_SRC = """
#version 120
varying vec2 vUV;
uniform sampler2D uTex;
void main() {
    gl_FragColor = texture2D(uTex, vUV);
}
"""


def _compile_shader(src: str, kind: int) -> int:
    s = glCreateShader(kind)
    glShaderSource(s, src)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s


def _link_program(vert_src: str, frag_src: str) -> int:
    vs = _compile_shader(vert_src, GL_VERTEX_SHADER)
    fs = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    glDeleteShader(vs)
    glDeleteShader(fs)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    return prog


def _ortho(l: float, r: float, b: float, t: float) -> np.ndarray:
    return np.array([
        [2.0 / (r - l), 0, 0, -(r + l) / (r - l)],
        [0, 2.0 / (t - b), 0, -(t + b) / (t - b)],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32).T


class _GpuBuffer:
    def __init__(self):
        self.vbo = glGenBuffers(1)
        self._cap = 0

    def upload(self, data: np.ndarray):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        n = data.nbytes
        if n > self._cap:
            glBufferData(GL_ARRAY_BUFFER, n, data, GL_DYNAMIC_DRAW)
            self._cap = n
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, n, data)


class _GLRenderer:
    def __init__(self):
        self._prog = _link_program(_VERT_SRC, _FRAG_SRC)
        self._prog_circ = _link_program(_VERT_CIRCLE_SRC, _FRAG_CIRCLE_SRC)
        self._prog_tex = _link_program(_VERT_TEX_SRC, _FRAG_TEX_SRC)
        self._buf = _GpuBuffer()
        self._unit_circle = self._make_unit_fan(32)
        self._tex_buf = _GpuBuffer()
        self._text_textures: dict = {}

    def _make_unit_fan(self, n: int) -> np.ndarray:
        pts = [[0.0, 0.0]]
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            pts.append([math.cos(a), math.sin(a)])
        return np.array(pts, dtype=np.float32)

    def set_mvp(self, mvp: np.ndarray):
        glUseProgram(self._prog)
        glUniformMatrix4fv(self._get_loc(self._prog, "uMVP"), 1, GL_FALSE, mvp)
        glUseProgram(self._prog_circ)
        glUniformMatrix4fv(self._get_loc(self._prog_circ, "uMVP"), 1, GL_FALSE, mvp)
        glUseProgram(self._prog_tex)
        glUniformMatrix4fv(self._get_loc(self._prog_tex, "uMVP"), 1, GL_FALSE, mvp)

    def _get_loc(self, prog: int, name: str) -> int:
        from OpenGL.GL import glGetUniformLocation
        return glGetUniformLocation(prog, name)

    def _apply_pen_style(self, style: Qt.PenStyle):
        if style == Qt.PenStyle.SolidLine:
            glDisable(GL_LINE_STIPPLE)
        else:
            glEnable(GL_LINE_STIPPLE)
            if style == Qt.PenStyle.DashLine:
                glLineStipple(2, 0x00FF)
            elif style == Qt.PenStyle.DotLine:
                glLineStipple(1, 0xAAAA)
            elif style == Qt.PenStyle.DashDotLine:
                glLineStipple(1, 0x1C47)
            elif style == Qt.PenStyle.DashDotDotLine:
                glLineStipple(1, 0x1C1C)

    def draw_polyline(self, pts: np.ndarray, color: Tuple, width: float = 1.0, style: Qt.PenStyle = Qt.PenStyle.SolidLine):
        if len(pts) < 2:
            return
        glUseProgram(self._prog)
        glUniform4f(self._get_loc(self._prog, "uColor"), *color)
        self._buf.upload(pts.astype(np.float32))

        self._apply_pen_style(style)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glLineWidth(max(1.0, width))
        glDrawArrays(GL_LINE_STRIP, 0, len(pts))
        glDisableVertexAttribArray(0)
        glDisable(GL_LINE_STIPPLE)

    def draw_lines(self, pts: np.ndarray, color: Tuple, width: float = 1.0, style: Qt.PenStyle = Qt.PenStyle.SolidLine):
        if len(pts) < 2:
            return
        glUseProgram(self._prog)
        glUniform4f(self._get_loc(self._prog, "uColor"), *color)
        self._buf.upload(pts.astype(np.float32))

        self._apply_pen_style(style)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glLineWidth(max(1.0, width))
        glDrawArrays(GL_LINES, 0, len(pts))
        glDisableVertexAttribArray(0)
        glDisable(GL_LINE_STIPPLE)

    def draw_dots(self, pts: np.ndarray, color: Tuple, size: float = 4.0):
        if len(pts) == 0:
            return
        glUseProgram(self._prog)
        glUniform4f(self._get_loc(self._prog, "uColor"), *color)
        self._buf.upload(pts.astype(np.float32))

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glPointSize(size)
        glDrawArrays(GL_POINTS, 0, len(pts))
        glDisableVertexAttribArray(0)

    def draw_circle_fill(self, cx: float, cy: float, r: float, color: Tuple):
        fan = self._unit_circle.copy()
        fan[:, 0] = fan[:, 0] * r + cx
        fan[:, 1] = fan[:, 1] * r + cy
        glUseProgram(self._prog)
        glUniform4f(self._get_loc(self._prog, "uColor"), *color)
        self._buf.upload(fan)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLE_FAN, 0, len(fan))
        glDisableVertexAttribArray(0)

    def draw_rect_fill(self, x: float, y: float, w: float, h: float, color: Tuple):
        pts = np.array([
            [x, y], [x + w, y], [x + w, y + h],
            [x, y], [x + w, y + h], [x, y + h],
        ], dtype=np.float32)
        glUseProgram(self._prog)
        glUniform4f(self._get_loc(self._prog, "uColor"), *color)
        self._buf.upload(pts)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)

    def draw_rect_outline(self, x: float, y: float, w: float, h: float, color: Tuple, lw: float = 1.0):
        pts = np.array([
            [x, y], [x + w, y], [x + w, y],
            [x + w, y + h], [x + w, y + h], [x, y + h],
            [x, y + h], [x, y],
        ], dtype=np.float32)
        self.draw_lines(pts, color, lw)


def _qcolor_to_gl(c: QColor, alpha_override: Optional[float] = None) -> Tuple[float, float, float, float]:
    a = alpha_override if alpha_override is not None else c.alphaF()
    return (c.redF(), c.greenF(), c.blueF(), a)


def _pts_to_screen(xs: List[float], ys: List[float],
                   x0: float, dx: float, y0: float, dy: float,
                   pl: float, pb: float, pw: float, ph: float) -> np.ndarray:
    if not xs:
        return np.empty((0, 2), dtype=np.float32)
    xa = np.asarray(xs, dtype=np.float64)
    ya = np.asarray(ys, dtype=np.float64)
    sx = pl + (xa - x0) / dx * pw
    sy = pb - (ya - y0) / dy * ph
    np.clip(sx, -_COORD_CLAMP, _COORD_CLAMP, out=sx)
    np.clip(sy, -_SCREEN_Y_CLAMP, _SCREEN_Y_CLAMP, out=sy)
    return np.column_stack([sx, sy]).astype(np.float32)


def _log_pts_to_screen(xs: List[float], ys: List[float],
                       x0: float, dx: float, y0: float, dy: float,
                       pl: float, pb: float, pw: float, ph: float,
                       log_x: bool, log_y: bool) -> np.ndarray:
    if not xs:
        return np.empty((0, 2), dtype=np.float32)
    xa = np.asarray(xs, dtype=np.float64)
    ya = np.asarray(ys, dtype=np.float64)
    if log_x:
        xa = np.where(xa > 0, np.log10(xa), -300.0)
    if log_y:
        ya = np.where(ya > 0, np.log10(ya), -300.0)
    sx = pl + (xa - x0) / dx * pw
    sy = pb - (ya - y0) / dy * ph
    np.clip(sx, -_COORD_CLAMP, _COORD_CLAMP, out=sx)
    np.clip(sy, -_SCREEN_Y_CLAMP, _SCREEN_Y_CLAMP, out=sy)
    return np.column_stack([sx, sy]).astype(np.float32)


class _GLPlotCanvas(QOpenGLWidget):
    def __init__(self, chart: "ChartWidget"):
        super().__init__(chart)
        self._chart = chart
        self._renderer: Optional[_GLRenderer] = None
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
        self._nearest_cache_key: Optional[tuple] = None
        self._nearest_cache_result = None
        self._dragging_ruler_pt: Optional[int] = None

        fmt = QSurfaceFormat()
        fmt.setSamples(8)
        fmt.setSwapInterval(1)
        self.setFormat(fmt)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu)

    def initializeGL(self):
        try:
            self._renderer = _GLRenderer()
            glEnable(GL_MULTISAMPLE)
            glEnable(GL_BLEND)
            glEnable(GL_POINT_SMOOTH)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        except Exception as e:
            print("GL init failed:", e)
            self._renderer = None

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)

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

    def _to_pt_log(self, xv, yv, x0, dx, y0, dy, pr, log_x: bool, log_y: bool) -> QPointF:
        lx = to_log(xv) if log_x else xv
        ly = to_log(yv) if log_y else yv
        return self._to_pt(lx, ly, x0, dx, y0, dy, pr)

    def _screen_to_data(self, sx, sy, x0, dx, y0, dy, pr, log_x, log_y):
        lx = x0 + (sx - pr.left()) / pr.width() * dx
        ly = y0 + (pr.bottom() - sy) / pr.height() * dy
        xv = 10 ** lx if log_x else lx
        yv = 10 ** ly if log_y else ly
        return xv, yv

    def _view_params(self, pr):
        c = self._chart
        x0 = to_log(c.vx0) if c.log_x else c.vx0
        x1 = to_log(c.vx1) if c.log_x else c.vx1
        y0 = to_log(c.vy0) if c.log_y else c.vy0
        y1 = to_log(c.vy1) if c.log_y else c.vy1
        dx = x1 - x0 if abs(x1 - x0) > 1e-300 else 1.0
        dy = y1 - y0 if abs(y1 - y0) > 1e-300 else 1.0
        return x0, x1, y0, y1, dx, dy

    def _nearest_on_segments(self, mouse, xs, ys, pr, x0, dx, y0, dy, log_x, log_y):
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
                                                          ((mouse.x() - pt0.x()) * seg_x + (
                                                                      mouse.y() - pt0.y()) * seg_y) / seg_len2))
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
            r = self._nearest_on_segments(mouse, dxs, dys, pr, x0, dx, y0, dy, log_x, log_y)
            if r and r[2] < best_d:
                best_d = r[2]
                best = (r[0], r[1], r[2], item)
        for item in c.fits:
            if not item.visible or not item.xs:
                continue
            r = self._nearest_on_segments(mouse, item.xs, item.ys, pr, x0, dx, y0, dy, log_x, log_y)
            if r and r[2] < best_d:
                best_d = r[2]
                best = (r[0], r[1], r[2], item)
        for item in c.functions:
            if not item.visible or not item.xs:
                continue
            fn_xs_f = [item.xs[i] for i in range(len(item.xs)) if item.ys[i] is not None]
            fn_ys_f = [item.ys[i] for i in range(len(item.ys)) if item.ys[i] is not None]
            if not fn_xs_f:
                continue
            r = self._nearest_on_segments(mouse, fn_xs_f, fn_ys_f, pr, x0, dx, y0, dy, log_x, log_y)
            if r and r[2] < best_d:
                best_d = r[2]
                best = (r[0], r[1], r[2], item)
        result = best if best and best[2] <= _SNAP_RADIUS_PX else None
        self._nearest_cache_key = key
        self._nearest_cache_result = result
        return result

    def _find_infline_at(self, pos, pr, x0, dx, y0, dy):
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

    def _find_ruler_handle_at(self, pos, pr, x0, dx, y0, dy):
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        for ruler in c.rulers:
            if not ruler.visible or not ruler.draggable:
                continue
            for idx, (xv, yv) in enumerate(((ruler.x0, ruler.y0), (ruler.x1, ruler.y1))):
                pt = self._to_pt_log(xv, yv, x0, dx, y0, dy, pr, log_x, log_y)
                if math.hypot(pt.x() - pos.x(), pt.y() - pos.y()) <= ruler.handle_radius + 4:
                    return (ruler, idx)
        return None

    def _find_scatter_point_at(self, pos, pr, x0, dx, y0, dy):
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

    def paintGL(self):
        if self._renderer is None:
            return

        p = QPainter(self)
        p.beginNativePainting()

        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClear(GL_COLOR_BUFFER_BIT)
        w, h = self.width(), self.height()
        mvp = _ortho(0, w, h, 0)
        self._renderer.set_mvp(mvp)
        pal = self.palette()
        bg = pal.window().color()
        fg = pal.windowText().color()
        bg_gl = _qcolor_to_gl(bg)
        fg_gl = _qcolor_to_gl(fg)

        self._renderer.draw_rect_fill(0, 0, w, h, bg_gl)
        c = self._chart
        pr = self._plot_rect()
        x0, x1, y0, y1, dx, dy = self._view_params(pr)
        log_x, log_y = c.log_x, c.log_y

        if not math.isfinite(dx) or abs(dx) < 1e-300:
            dx = 1.0
        if not math.isfinite(dy) or abs(dy) < 1e-300:
            dy = 1.0

        n_x = max(2, pr.width() // max(1, c.grid_px_x))
        n_y = max(2, pr.height() // max(1, c.grid_px_y))
        xt = nice_log_ticks(10 ** x0, 10 ** x1) if log_x else nice_ticks(x0, x1, n_x)
        xt_screen = [to_log(v) for v in xt] if log_x else xt
        yt = nice_log_ticks(10 ** y0, 10 ** y1) if log_y else nice_ticks(y0, y1, n_y)
        yt_screen = [to_log(v) for v in yt] if log_y else yt

        gr_col = (*fg_gl[:3], 0.18)
        grid_lines = []
        for tv in xt_screen:
            sx = pr.left() + (tv - x0) / dx * pr.width()
            grid_lines += [[sx, pr.top()], [sx, pr.bottom()]]
        for tv in yt_screen:
            sy = pr.bottom() - (tv - y0) / dy * pr.height()
            grid_lines += [[pr.left(), sy], [pr.right(), sy]]
        if grid_lines:
            self._renderer.draw_lines(np.array(grid_lines, dtype=np.float32), gr_col, 1.0)

        ax_col = (*fg_gl[:3], 0.32)
        self._renderer.draw_rect_outline(pr.left(), pr.top(), pr.width(), pr.height(), ax_col, 1.0)

        if self._range_sel_x is not None:
            rx_lo, rx_hi = sorted(self._range_sel_x)
            lx_lo = to_log(rx_lo) if log_x else rx_lo
            lx_hi = to_log(rx_hi) if log_x else rx_hi
            sx_lo = max(float(pr.left()), pr.left() + (lx_lo - x0) / dx * pr.width())
            sx_hi = min(float(pr.right()), pr.left() + (lx_hi - x0) / dx * pr.width())
            if sx_hi > sx_lo:
                sel_col = (*fg_gl[:3], _RANGE_SEL_ALPHA / 255.0)
                self._renderer.draw_rect_fill(sx_lo, pr.top(), sx_hi - sx_lo, pr.height(), sel_col)
                sel_border = (*fg_gl[:3], 0.4)
                self._renderer.draw_lines(np.array([
                    [sx_lo, pr.top()], [sx_lo, pr.bottom()],
                    [sx_hi, pr.top()], [sx_hi, pr.bottom()],
                ], dtype=np.float32), sel_border, 1.0)

        pl, pb, pw, ph = float(pr.left()), float(pr.bottom()), float(pr.width()), float(pr.height())
        glEnable(GL_SCISSOR_TEST)
        glScissor(pr.left(), h - pr.bottom() - 1, pr.width(), pr.height() + 1)

        for ln in c.inflines:
            if not ln.visible:
                continue
            lc = _qcolor_to_gl(ln.pen.color())
            if ln.horizontal:
                ly = to_log(ln.value) if log_y else ln.value
                sy = pb - (ly - y0) / dy * ph
                self._renderer.draw_lines(np.array([[pl, sy], [pl + pw, sy]], dtype=np.float32), lc,
                                          ln.pen.widthF() or 1.0, ln.pen.style())
            else:
                lx = to_log(ln.value) if log_x else ln.value
                sx = pl + (lx - x0) / dx * pw
                self._renderer.draw_lines(np.array([[sx, pr.top()], [sx, pb]], dtype=np.float32), lc,
                                          ln.pen.widthF() or 1.0, ln.pen.style())

        x_lo = min(c.vx0, c.vx1)
        x_hi = max(c.vx0, c.vx1)
        x_lo_fn = x_lo - (x_hi - x_lo) * 0.05
        x_hi_fn = x_hi + (x_hi - x_lo) * 0.05

        for fit in c.fits:
            if not fit.visible:
                continue
            fit._recompute(x_lo, x_hi, threaded=c._threaded_fit)
            if len(fit.xs) < 2:
                continue
            pts = _log_pts_to_screen(fit.xs, fit.ys, x0, dx, y0, dy, pl, pb, pw, ph, log_x, log_y)
            fc = _qcolor_to_gl(fit.pen.color())
            lw = fit.pen.widthF() or 1.0
            self._renderer.draw_polyline(pts, fc, lw, fit.pen.style())

        for fn_item in c.functions:
            if not fn_item.visible:
                continue
            fn_xs, fn_ys = fn_item.evaluate(x_lo_fn, x_hi_fn, max(1, pr.width()), max(1, pr.height()))
            if len(fn_xs) < 2:
                continue
            valid_pairs = [(x, y) for x, y in zip(fn_xs, fn_ys) if y is not None]
            if len(valid_pairs) < 2:
                continue
            vxs = [p[0] for p in valid_pairs]
            vys = [p[1] for p in valid_pairs]
            pts = _log_pts_to_screen(vxs, vys, x0, dx, y0, dy, pl, pb, pw, ph, log_x, log_y)
            fc = _qcolor_to_gl(fn_item.pen.color())
            lw = fn_item.pen.widthF() or 1.0
            self._renderer.draw_polyline(pts, fc, lw, fn_item.pen.style())

        for item in c.lines:
            if not item.visible or not item.raw_visible or len(item.xs) < 2:
                continue
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            pts = _log_pts_to_screen(dxs, dys, x0, dx, y0, dy, pl, pb, pw, ph, log_x, log_y)
            lc = _qcolor_to_gl(item.pen.color())
            lw = item.pen.widthF() or 1.0

            if item.fill_under and len(pts) >= 2:
                baseline_y = pb - (0.0 - y0) / dy * ph
                baseline_y = max(float(pr.top()), min(float(pb), baseline_y))
                fill_pts = np.vstack([
                    pts,
                    [[pts[-1, 0], baseline_y], [pts[0, 0], baseline_y]],
                ])
                fill_col = (*lc[:3], item.fill_alpha / 255.0)
                fan_pts = np.vstack([[[pts[0, 0], baseline_y]], fill_pts])
                glUseProgram(self._renderer._prog)
                from OpenGL.GL import glGetUniformLocation, glUniform4f as _u4f
                _u4f(glGetUniformLocation(self._renderer._prog, "uColor"), *fill_col)
                self._renderer._buf.upload(fan_pts)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
                glDrawArrays(GL_TRIANGLE_FAN, 0, len(fan_pts))
                glDisableVertexAttribArray(0)

            self._renderer.draw_polyline(pts, lc, lw, item.pen.style())

        for item in c.lines_r2:
            if not item.visible or not item.raw_visible or len(item.xs) < 2:
                continue
            y0r = to_log(c.vy0_r) if log_y else c.vy0_r
            y1r = to_log(c.vy1_r) if log_y else c.vy1_r
            dyr = y1r - y0r if abs(y1r - y0r) > 1e-300 else 1.0
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            pts = _log_pts_to_screen(dxs, dys, x0, dx, y0r, dyr, pl, pb, pw, ph, log_x, log_y)
            lc = _qcolor_to_gl(item.pen.color())
            self._renderer.draw_polyline(pts, lc, item.pen.widthF() or 1.0, item.pen.style())

        all_scatters = list(c.scatters) + list(c.scatters_r2)
        for item in all_scatters:
            if not item.visible or not item.raw_visible or not item.xs:
                continue
            is_r2 = item in c.scatters_r2
            y0_eff = (to_log(c.vy0_r) if log_y else c.vy0_r) if is_r2 else y0
            dy_eff_v = ((to_log(c.vy1_r) - to_log(c.vy0_r)) if log_y else (c.vy1_r - c.vy0_r)) if is_r2 else dy
            if is_r2 and abs(dy_eff_v) < 1e-300:
                dy_eff_v = 1.0
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            pts = _log_pts_to_screen(dxs, dys, x0, dx, y0_eff, dy_eff_v, pl, pb, pw, ph, log_x, log_y)
            r = item.size / 2.0
            ic = _qcolor_to_gl(item.color)
            for i in range(len(pts)):
                cx, cy = float(pts[i, 0]), float(pts[i, 1])
                is_sel = item.selected_idx is not None and i == item.selected_idx
                if is_sel:
                    hi_col = (*ic[:3], 0.31)
                    self._renderer.draw_circle_fill(cx, cy, r * 2.5, hi_col)
                self._renderer.draw_circle_fill(cx, cy, r, ic)

        glDisable(GL_SCISSOR_TEST)
        if self._mouse_pos is not None and not self._rb_active:
            self._paint_crosshair_gl(pr, x0, dx, y0, dy, fg_gl, bg_gl)

        if self._rb_active and self._rb_start and self._rb_end:
            rb = QRectF(self._rb_start, self._rb_end).normalized()
            rb_col = (*fg_gl[:3], 0.16)
            rb_border = (*fg_gl[:3], 0.63)
            self._renderer.draw_rect_fill(rb.left(), rb.top(), rb.width(), rb.height(), rb_col)
            self._renderer.draw_rect_outline(rb.left(), rb.top(), rb.width(), rb.height(), rb_border, 1.0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

        p.endNativePainting()

        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = QFontMetrics(c.font)
        p.setFont(c.font)
        lb_col = QColor(fg)
        lb_col.setAlpha(255)
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

        if self._has_right_axis():
            y0r = to_log(c.vy0_r) if c.log_y else c.vy0_r
            y1r = to_log(c.vy1_r) if c.log_y else c.vy1_r
            dyr = y1r - y0r if abs(y1r - y0r) > 1e-300 else 1.0
            ytr = nice_log_ticks(10 ** y0r, 10 ** y1r) if log_y else nice_ticks(y0r, y1r, max(2, pr.height() // max(1,
                                                                                                                    c.grid_px_y)))
            ytr_screen = [to_log(v) for v in ytr] if log_y else ytr
            r2_col = QColor(fg)
            r2_col.setAlpha(160)
            p.setPen(r2_col)
            for tv, ts in zip(ytr, ytr_screen):
                sy = int(pr.bottom() - (ts - y0r) / dyr * pr.height())
                lbl = fmt(tv)
                p.drawText(pr.right() + 6, sy + fm.ascent() // 2, lbl)
            if c.label_right:
                p.save()
                p.translate(self.width() - 8, pr.top() + pr.height() // 2)
                p.rotate(90)
                lw_txt = fm.horizontalAdvance(c.label_right)
                p.drawText(-lw_txt // 2, fm.ascent() // 2, c.label_right)
                p.restore()

        if c.label_bottom:
            lw_txt = fm.horizontalAdvance(c.label_bottom)
            p.drawText(pr.left() + (pr.width() - lw_txt) // 2, self.height() - 3, c.label_bottom)

        if c.label_left:
            p.save()
            p.translate(11, pr.top() + pr.height() // 2)
            p.rotate(-90)
            lw_txt = fm.horizontalAdvance(c.label_left)
            p.drawText(-lw_txt // 2, fm.ascent() // 2, c.label_left)
            p.restore()

        for fit in c.fits:
            if fit.visible and fit.show_formula and len(fit.xs) >= 2:
                formula = fit.getFormula()
                if formula:
                    pts = _log_pts_to_screen(fit.xs, fit.ys, x0, dx, y0, dy, pl, pb, pw, ph, log_x, log_y)
                    mid_idx = len(pts) // 2
                    mid_pt = QPointF(float(pts[mid_idx, 0]), float(pts[mid_idx, 1]))
                    self._draw_formula_tag_qt(p, mid_pt, formula, fg, bg, fm)

        if self._mouse_pos is not None and not self._rb_active:
            self._paint_crosshair_overlay(p, pr, x0, dx, y0, dy, fm, fg, bg)

        if self._show_latest:
            self._paint_latest_points_qt(p, pr, x0, dx, y0, dy, fm, fg, bg)

        if self._show_analytics:
            self._paint_analytics_qt(p, pr, fg, bg, fm)

        if c.show_legend:
            self._paint_legend_qt(p, pr, fg, bg, fm)

        for ruler in c.rulers:
            if ruler.visible:
                self._paint_ruler_qt(p, pr, x0, dx, y0, dy, ruler, fg, bg, fm)

        p.end()

    def _paint_crosshair_overlay(self, p, pr, x0, dx, y0, dy, fm, fg, bg):
        mp = self._mouse_pos
        c = self._chart
        nearest = self._find_nearest(mp, pr, x0, dx, y0, dy)
        if nearest is None:
            return
        xi, yi, _, item = nearest
        snap = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, c.log_x, c.log_y)
        self._paint_tooltip_qt(p, pr, xi, yi, snap, fg, bg, fm)

    def _paint_crosshair_gl(self, pr, x0, dx, y0, dy, fg_gl, bg_gl):
        mp = self._mouse_pos
        c = self._chart
        nearest = self._find_nearest(mp, pr, x0, dx, y0, dy)
        ch_col = (*fg_gl[:3], 0.31)
        ch_pts = np.array([
            [mp.x(), pr.top()], [mp.x(), pr.bottom()],
            [pr.left(), mp.y()], [pr.right(), mp.y()],
        ], dtype=np.float32)
        self._renderer.draw_lines(ch_pts, ch_col, 1.0)
        if nearest is None:
            return
        xi, yi, _, item = nearest
        snap = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, c.log_x, c.log_y)
        dot_col = (*fg_gl[:3], 0.86)
        self._renderer.draw_circle_fill(snap.x(), snap.y(), _SNAP_DOT_R, dot_col)

    def _paint_tooltip_qt(self, p, pr, xi, yi, snap, fg, bg, fm):
        lx = f"{tr('chart_widget.tooltip_x')}: {fmt(xi)}"
        ly = f"{tr('chart_widget.tooltip_y')}: {fmt(yi)}"
        tw = max(fm.horizontalAdvance(lx), fm.horizontalAdvance(ly)) + 16
        th = fm.height() * 2 + 12
        tx = int(snap.x()) + _TOOLTIP_MARGIN
        ty = int(snap.y()) - th - 4
        if tx + tw > pr.right():
            tx = int(snap.x()) - tw - _TOOLTIP_MARGIN
        if ty < pr.top():
            ty = int(snap.y()) + 8
        bg_ = QColor(bg)
        bg_.setAlpha(215)
        br_ = QColor(fg)
        br_.setAlpha(100)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(br_, 1))
        p.drawRoundedRect(tx, ty, tw, th, 4, 4)
        p.setPen(fg)
        p.drawText(tx + 8, ty + fm.ascent() + 4, lx)
        p.drawText(tx + 8, ty + fm.ascent() + 4 + fm.height(), ly)

    def _draw_formula_tag_qt(self, p, pt, formula, fg, bg, fm):
        tw = fm.horizontalAdvance(formula) + 10
        th = fm.height() + 6
        tx = int(pt.x()) - tw // 2
        ty = int(pt.y()) - th - 4
        bg_ = QColor(bg)
        bg_.setAlpha(200)
        br_ = QColor(fg)
        br_.setAlpha(80)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(br_, 1))
        p.drawRoundedRect(tx, ty, tw, th, 3, 3)
        p.setPen(fg)
        p.drawText(tx + 5, ty + fm.ascent() + 3, formula)

    def _paint_latest_points_qt(self, p, pr, x0, dx, y0, dy, fm, fg, bg):
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        pad = _LATEST_TAG_PAD
        rnd = _LATEST_TAG_ROUND
        th = fm.height() + pad * 2
        from .palette import contrast_color
        for item in c.lines:
            if not item.visible or not item.xs:
                continue
            xi, yi = item.xs[-1], item.ys[-1]
            snap = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, log_x, log_y)
            sx, sy = snap.x(), snap.y()
            color = item.pen.color()
            x_lbl = fmt(xi)
            x_tw = fm.horizontalAdvance(x_lbl) + pad * 2
            xtx = int(sx) - x_tw // 2
            xty = pr.bottom() + 2
            if pr.left() - x_tw <= sx <= pr.right() + x_tw:
                txt_col = contrast_color(color)
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
                txt_col = contrast_color(color)
                p.setBrush(QBrush(color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(ytx, yty, y_tw, th, rnd, rnd)
                p.setPen(txt_col)
                p.drawText(ytx + pad, yty + fm.ascent(), y_lbl)

    def _paint_analytics_qt(self, p, pr, fg, bg, fm):
        import statistics
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
        table = []
        for _, it in named:
            if range_lo is not None and range_hi is not None:
                pairs = [(x, y) for x, y in zip(it.xs, it.ys) if range_lo <= x <= range_hi]
                fxs, fys = [p[0] for p in pairs], [p[1] for p in pairs]
            else:
                fxs, fys = it.xs, it.ys
            n = len(fxs)
            if n == 0:
                table.append(["-"] * len(row_keys))
                continue
            st = fmt(statistics.stdev(fys)) if n > 1 else "-"
            table.append([
                str(n), fmt(min(fxs)), fmt(max(fxs)),
                fmt(min(fys)), fmt(max(fys)),
                fmt(statistics.mean(fys)), st,
                fmt(trapezoid_integral(fxs, fys)),
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
        bg_ = QColor(bg)
        bg_.setAlpha(210)
        brd_ = QColor(fg)
        brd_.setAlpha(70)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(brd_, 1))
        p.drawRoundedRect(ax, ay, total_w, total_h, 4, 4)
        bold_f = QFont(c.font)
        bold_f.setBold(True)
        hdr_col = QColor(fg)
        hdr_col.setAlpha(220)
        lbl_col = QColor(fg)
        lbl_col.setAlpha(150)
        for ci, (name, _) in enumerate(named):
            cx = ax + pad + lbl_w + ci * val_w + val_w // 2
            p.setFont(bold_f)
            p.setPen(hdr_col)
            p.drawText(cx - fm.horizontalAdvance(name) // 2, ay + pad + fm.ascent(), name)
        p.setFont(c.font)
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

    def _paint_legend_qt(self, p, pr, fg, bg, fm):
        c = self._chart
        entries = []
        for i, it in enumerate(c.lines + c.lines_r2):
            if not it.visible:
                continue
            suffix = " (R)" if it in c.lines_r2 else ""
            label = it.label or (tr("chart_widget.legend_label", n=i + 1) + suffix)
            entries.append((label, it.pen.color(), False))
        offset = len(c.lines) + len(c.lines_r2)
        for i, it in enumerate(c.scatters + c.scatters_r2):
            if not it.visible:
                continue
            suffix = " (R)" if it in c.scatters_r2 else ""
            label = it.label or (tr("chart_widget.legend_label", n=offset + i + 1) + suffix)
            color = it.color if hasattr(it, 'color') else QColor(255, 255, 255)
            entries.append((label, color, True))
        for i, fit in enumerate(c.fits):
            if not fit.visible:
                continue
            entries.append((fit.label or tr("chart_widget.analytics_fit", n=i + 1), fit.pen.color(), False))
        for i, fn_item in enumerate(c.functions):
            if not fn_item.visible:
                continue
            entries.append((fn_item.label or f"f(x) {i + 1}", fn_item.pen.color(), False))
        if not entries:
            return
        pad = _LEGEND_PAD
        sw = _LEGEND_SWATCH
        row_h = max(fm.height(), sw) + 4
        max_w = max(fm.horizontalAdvance(e[0]) for e in entries) + sw + pad * 2 + 6
        total_h = pad * 2 + row_h * len(entries)
        lx = pr.right() - max_w - 4
        ly = pr.bottom() - total_h - 4
        bg_ = QColor(bg)
        bg_.setAlpha(200)
        brd_ = QColor(fg)
        brd_.setAlpha(60)
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(brd_, 1))
        p.drawRoundedRect(lx, ly, max_w, total_h, 4, 4)
        p.setFont(c.font)
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
            txt_col = QColor(fg)
            txt_col.setAlpha(210)
            p.setPen(txt_col)
            p.drawText(lx + pad + sw + 6, ry + fm.ascent(), label)

    def _paint_ruler_qt(self, p, pr, x0, dx, y0, dy, ruler, fg, bg, fm):
        c = self._chart
        log_x, log_y = c.log_x, c.log_y
        pt0 = self._to_pt_log(ruler.x0, ruler.y0, x0, dx, y0, dy, pr, log_x, log_y)
        pt1 = self._to_pt_log(ruler.x1, ruler.y1, x0, dx, y0, dy, pr, log_x, log_y)
        p.setPen(ruler.pen)
        p.drawLine(pt0, pt1)
        r = ruler.handle_radius
        for pt in (pt0, pt1):
            p.setBrush(QBrush(ruler.pen.color()))
            p.drawEllipse(pt, r, r)
        dist_lbl = fmt(ruler.distance)
        mid = QPointF((pt0.x() + pt1.x()) / 2, (pt0.y() + pt1.y()) / 2)
        bg_ = QColor(bg)
        bg_.setAlpha(190)
        brd_ = QColor(fg)
        brd_.setAlpha(80)
        tw = fm.horizontalAdvance(dist_lbl) + 8
        th = fm.height() + 4
        p.setBrush(QBrush(bg_))
        p.setPen(QPen(brd_, 1))
        p.drawRoundedRect(int(mid.x()) - tw // 2, int(mid.y()) - th // 2, tw, th, 3, 3)
        p.setPen(fg)
        p.drawText(int(mid.x()) - tw // 2 + 4, int(mid.y()) + fm.ascent() // 2, dist_lbl)

    def grab_image(self) -> QPixmap:
        return self.grab()

    def contextMenuEvent(self, ev):
        from PyQt5.QtWidgets import QMenu, QAction, QActionGroup
        c = self._chart
        menu = QMenu(self)
        act_autofit = QAction(tr("chart_widget.btn_autofit_toggle"), menu)
        act_autofit.setCheckable(True)
        act_autofit.setChecked(c._autofit)
        act_autofit.triggered.connect(lambda v: c.setAutofit(v))
        menu.addAction(act_autofit)
        act_latest = QAction(tr("chart_widget.btn_latest"), menu)
        act_latest.setCheckable(True)
        act_latest.setChecked(self._show_latest)
        act_latest.triggered.connect(self.toggleLatestPoint)
        menu.addAction(act_latest)
        act_legend = QAction(tr("chart_widget.ctx_legend"), menu)
        act_legend.setCheckable(True)
        act_legend.setChecked(c.show_legend)
        act_legend.triggered.connect(lambda v: setattr(c, "show_legend", v) or self.update())
        menu.addAction(act_legend)
        menu.addSeparator()
        act_log_x = QAction(tr("chart_widget.ctx_log_scale") + " X", menu)
        act_log_x.setCheckable(True)
        act_log_x.setChecked(c.log_x)
        act_log_x.triggered.connect(lambda v: c.setLogScale(v, c.log_y))
        menu.addAction(act_log_x)
        act_log_y = QAction(tr("chart_widget.ctx_log_scale") + " Y", menu)
        act_log_y.setCheckable(True)
        act_log_y.setChecked(c.log_y)
        act_log_y.triggered.connect(lambda v: c.setLogScale(c.log_x, v))
        menu.addAction(act_log_y)
        menu.addSeparator()
        zoom_menu = menu.addMenu(tr("chart_widget.ctx_zoom_lock"))
        grp_z = QActionGroup(zoom_menu)
        grp_z.setExclusive(True)
        for key, lbl in (("both", tr("chart_widget.ctx_zoom_both")), ("x", "X"), ("y", "Y")):
            act = QAction(lbl, zoom_menu)
            act.setCheckable(True)
            act.setChecked(self._zoom_lock == key)
            act.triggered.connect(lambda checked, k=key: self.setZoomLock(k))
            zoom_menu.addAction(act)
            grp_z.addAction(act)
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
            nvx0 = cx_log + (x0 - cx_log) * factor
            nvx1 = cx_log + (x1 - cx_log) * factor
            sx = abs(nvx1 - nvx0)
            if math.isfinite(sx) and _ZOOM_MIN_SPAN <= sx <= _ZOOM_MAX_SPAN:
                c._vx0 = 10 ** nvx0 if c.log_x else nvx0
                c._vx1 = 10 ** nvx1 if c.log_x else nvx1
        if self._zoom_lock in ("both", "y"):
            nvy0 = cy_log + (y0 - cy_log) * factor
            nvy1 = cy_log + (y1 - cy_log) * factor
            sy = abs(nvy1 - nvy0)
            if math.isfinite(sy) and _ZOOM_MIN_SPAN <= sy <= _ZOOM_MAX_SPAN:
                c._vy0 = 10 ** nvy0 if c.log_y else nvy0
                c._vy1 = 10 ** nvy1 if c.log_y else nvy1
        self.update()
        ev.accept()

    def mousePressEvent(self, ev: QMouseEvent):
        pr = self._plot_rect()
        c = self._chart
        x0, x1, y0, y1, dx, dy = self._view_params(pr)
        if ev.button() == Qt.MouseButton.LeftButton:
            hit_ruler = self._find_ruler_handle_at(QPointF(ev.pos()), pr, x0, dx, y0, dy)
            if hit_ruler:
                self._dragging_ruler_pt = hit_ruler
                self.setCursor(Qt.CursorShape.CrossCursor)
                ev.accept()
                return
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
        if self._dragging_ruler_pt is not None:
            x0, x1, y0, y1, dx, dy = self._view_params(pr)
            ruler, pt_idx = self._dragging_ruler_pt
            xv, yv = self._screen_to_data(ev.pos().x(), ev.pos().y(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
            if pt_idx == 0:
                ruler.x0, ruler.y0 = xv, yv
            else:
                ruler.x1, ruler.y1 = xv, yv
            if ruler.changed is not None:
                ruler.changed()
            self.update()
            return
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
            new_vx0 = (to_log(self._pan_vx0) if c.log_x else self._pan_vx0) - ddx
            new_vy0 = (to_log(self._pan_vy0) if c.log_y else self._pan_vy0) + ddy
            if c.log_x:
                c._vx0 = 10 ** new_vx0
                c._vx1 = 10 ** (new_vx0 + dx)
            else:
                c._vx0 = new_vx0
                c._vx1 = new_vx0 + dx
            if c.log_y:
                c._vy0 = 10 ** new_vy0
                c._vy1 = 10 ** (new_vy0 + dy)
            else:
                c._vy0 = new_vy0
                c._vy1 = new_vy0 + dy
            self._nearest_cache_key = None
            self.update()
            return
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
        if self._dragging_ruler_pt is not None:
            self._dragging_ruler_pt = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        if self._dragging_infline is not None:
            self._dragging_infline = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        if self._rb_active and self._rb_start and self._rb_end:
            rb = QRectF(self._rb_start, self._rb_end).normalized()
            if rb.width() >= _RUBBERBAND_MIN_PX and rb.height() >= _RUBBERBAND_MIN_PX:
                pr = self._plot_rect()
                x0, x1, y0, y1, dx, dy = self._view_params(pr)
                c = self._chart
                x_lo_v, _ = self._screen_to_data(rb.left(), rb.top(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
                x_hi_v, _ = self._screen_to_data(rb.right(), rb.bottom(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
                _, y_hi_v = self._screen_to_data(rb.left(), rb.top(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
                _, y_lo_v = self._screen_to_data(rb.left(), rb.bottom(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
                c._vx0, c._vx1 = min(x_lo_v, x_hi_v), max(x_lo_v, x_hi_v)
                c._vy0, c._vy1 = min(y_lo_v, y_hi_v), max(y_lo_v, y_hi_v)
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
            xv, _ = self._screen_to_data(ev.pos().x(), ev.pos().y(), x0, dx, y0, dy, pr, c.log_x, c.log_y)
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