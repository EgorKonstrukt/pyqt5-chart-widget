"""OpenGL canvas backend."""
from __future__ import annotations
import math
import ctypes
from typing import List, Optional, Tuple, TYPE_CHECKING
from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                         QFontMetrics, QPainterPath, QPixmap,
                         QWheelEvent, QMouseEvent, QSurfaceFormat)
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
    GL_LINES, GL_LINE_STRIP, GL_POINTS, GL_TRIANGLES, GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP,
    GL_FLOAT, GL_FALSE, GL_TRUE,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    GL_COMPILE_STATUS, GL_LINK_STATUS,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW,
    GL_TEXTURE_2D, GL_TEXTURE0, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
    GL_SCISSOR_TEST, GL_MULTISAMPLE, GL_POINT_SMOOTH,
    glGetUniformLocation,
)
import numpy as np
from .canvas_base import (CanvasBase, _fmt_axis,
                           _COORD_CLAMP, _SCREEN_Y_CLAMP, _DECIMATE_THRESHOLD,
                           _RANGE_SEL_ALPHA, _RUBBERBAND_MIN_PX,
                           _SNAP_DOT_R, _TANGENT_HALF_FRAC,
                           _ANALYTICS_PAD, _ANALYTICS_ROW_H, _ANALYTICS_MAX_SERIES,
                           _TOOLTIP_MARGIN, _LEGEND_PAD, _LEGEND_SWATCH,
                           _LATEST_TAG_PAD, _LATEST_TAG_ROUND)
from .math_utils import (nice_ticks, nice_log_ticks, to_log, decimated,
                         fmt, get_fit_modes, trapezoid_integral)
from .items import _LineItem, _ScatterItem, _FitItem, _InfLine, _FunctionItem, _RulerItem
from .i18n import tr

if TYPE_CHECKING:
    from .chart_widget import ChartWidget

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

_CIRCLE_SEGMENTS = 32
_STIPPLE_PATTERNS = {
    Qt.PenStyle.DashLine: (2, 0x00FF),
    Qt.PenStyle.DotLine: (1, 0xAAAA),
    Qt.PenStyle.DashDotLine: (1, 0x1C47),
    Qt.PenStyle.DashDotDotLine: (1, 0x1C1C),
}


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


def _qcolor_to_gl(c: QColor, alpha_override: Optional[float] = None) -> Tuple[float, float, float, float]:
    a = alpha_override if alpha_override is not None else c.alphaF()
    return (c.redF(), c.greenF(), c.blueF(), a)


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


def _make_unit_fan(n: int) -> np.ndarray:
    """Build unit circle as triangle-fan vertices (center + n+1 rim points)."""
    angles = np.linspace(0, 2 * math.pi, n + 1, endpoint=True)
    rim = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    return np.vstack([np.zeros((1, 2), dtype=np.float32), rim])


_UNIT_FAN = _make_unit_fan(_CIRCLE_SEGMENTS)


def _build_fill_strip(pts: np.ndarray, baseline_y: float) -> np.ndarray:
    """Build triangle-strip vertices for fill-under a polyline."""
    n = len(pts)
    strip = np.empty((n * 2, 2), dtype=np.float32)
    strip[0::2] = pts
    strip[1::2, 0] = pts[:, 0]
    strip[1::2, 1] = baseline_y
    return strip


class _GpuBuffer:
    def __init__(self):
        self.vbo = glGenBuffers(1)
        self._cap = 0

    def upload(self, data: np.ndarray):
        """Upload numpy array to VBO, growing capacity as needed."""
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
        self._tex_buf = _GpuBuffer()
        self._text_textures: dict = {}
        self._uloc: dict[int, dict[str, int]] = {}
        self._last_prog = -1
        self._last_style = Qt.PenStyle.SolidLine

    def _loc(self, prog: int, name: str) -> int:
        """Return cached uniform location."""
        prg_locs = self._uloc.get(prog)
        if prg_locs is None:
            self._uloc[prog] = prg_locs = {}
        loc = prg_locs.get(name, -2)
        if loc == -2:
            loc = glGetUniformLocation(prog, name)
            prg_locs[name] = loc
        return loc

    def _use(self, prog: int):
        """Switch program only if different from current."""
        if prog != self._last_prog:
            glUseProgram(prog)
            self._last_prog = prog

    def set_mvp(self, mvp: np.ndarray):
        """Upload MVP matrix to all programs."""
        for prog in (self._prog, self._prog_circ, self._prog_tex):
            self._use(prog)
            glUniformMatrix4fv(self._loc(prog, "uMVP"), 1, GL_FALSE, mvp)

    def _apply_pen_style(self, style: Qt.PenStyle):
        if style == self._last_style:
            return
        self._last_style = style
        if style == Qt.PenStyle.SolidLine:
            glDisable(GL_LINE_STIPPLE)
        else:
            glEnable(GL_LINE_STIPPLE)
            factor, pattern = _STIPPLE_PATTERNS.get(style, (1, 0xFFFF))
            glLineStipple(factor, pattern)

    def _draw_arrays(self, pts: np.ndarray, mode: int, color: Tuple,
                     width: float = 1.0, style: Qt.PenStyle = Qt.PenStyle.SolidLine):
        if not len(pts):
            return
        self._use(self._prog)
        glUniform4f(self._loc(self._prog, "uColor"), *color)
        data = pts if pts.dtype == np.float32 else pts.astype(np.float32)
        self._buf.upload(data)
        self._apply_pen_style(style)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glLineWidth(max(1.0, width))
        glDrawArrays(mode, 0, len(pts))
        glDisableVertexAttribArray(0)
        if style != Qt.PenStyle.SolidLine:
            glDisable(GL_LINE_STIPPLE)
            self._last_style = Qt.PenStyle.SolidLine

    def draw_polyline(self, pts: np.ndarray, color: Tuple, width: float = 1.0,
                      style: Qt.PenStyle = Qt.PenStyle.SolidLine):
        """Draw a connected line strip."""
        if len(pts) < 2:
            return
        self._draw_arrays(pts, GL_LINE_STRIP, color, width, style)

    def draw_lines(self, pts: np.ndarray, color: Tuple, width: float = 1.0,
                   style: Qt.PenStyle = Qt.PenStyle.SolidLine):
        """Draw pairs of points as separate line segments."""
        if len(pts) < 2:
            return
        self._draw_arrays(pts, GL_LINES, color, width, style)

    def draw_dots(self, pts: np.ndarray, color: Tuple, size: float = 4.0):
        """Draw individual points."""
        if not len(pts):
            return
        self._use(self._prog)
        glUniform4f(self._loc(self._prog, "uColor"), *color)
        data = pts if pts.dtype == np.float32 else pts.astype(np.float32)
        self._buf.upload(data)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glPointSize(size)
        glDrawArrays(GL_POINTS, 0, len(pts))
        glDisableVertexAttribArray(0)

    def draw_circle_fill(self, cx: float, cy: float, r: float, color: Tuple):
        """Draw a single filled anti-aliased circle."""
        fan = _UNIT_FAN.copy()
        fan[:, 0] = fan[:, 0] * r + cx
        fan[:, 1] = fan[:, 1] * r + cy
        self._use(self._prog)
        glUniform4f(self._loc(self._prog, "uColor"), *color)
        self._buf.upload(fan)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLE_FAN, 0, len(fan))
        glDisableVertexAttribArray(0)

    def draw_circles_batch(self, centers: np.ndarray, r: float, color: Tuple,
                           sel_idx: Optional[int] = None,
                           sel_r_mul: float = 2.5, sel_alpha: float = 0.31):
        """Draw multiple circles by tiling fan geometry into a single VBO."""
        if not len(centers):
            return
        fan_local = _UNIT_FAN
        n_fan = len(fan_local)
        n_pts = len(centers)
        self._use(self._prog)
        glEnableVertexAttribArray(0)
        if sel_idx is not None:
            cx, cy = float(centers[sel_idx, 0]), float(centers[sel_idx, 1])
            halo = np.empty((n_fan, 2), dtype=np.float32)
            halo[:, 0] = fan_local[:, 0] * r * sel_r_mul + cx
            halo[:, 1] = fan_local[:, 1] * r * sel_r_mul + cy
            glUniform4f(self._loc(self._prog, "uColor"), *color[:3], sel_alpha)
            self._buf.upload(halo)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
            glDrawArrays(GL_TRIANGLE_FAN, 0, n_fan)
            glUniform4f(self._loc(self._prog, "uColor"), *color)
        verts = np.empty((n_pts * n_fan, 2), dtype=np.float32)
        for i in range(n_pts):
            base = i * n_fan
            verts[base:base + n_fan, 0] = fan_local[:, 0] * r + centers[i, 0]
            verts[base:base + n_fan, 1] = fan_local[:, 1] * r + centers[i, 1]
        self._buf.upload(verts)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        for i in range(n_pts):
            glDrawArrays(GL_TRIANGLE_FAN, i * n_fan, n_fan)
        glDisableVertexAttribArray(0)

    def draw_fill_under(self, pts: np.ndarray, baseline_y: float, color: Tuple):
        """Draw filled area between polyline and baseline using GL_TRIANGLE_STRIP."""
        if len(pts) < 2:
            return
        strip = _build_fill_strip(pts, baseline_y)
        self._use(self._prog)
        glUniform4f(self._loc(self._prog, "uColor"), *color)
        self._buf.upload(strip)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(strip))
        glDisableVertexAttribArray(0)

    def draw_rect_fill(self, x: float, y: float, w: float, h: float, color: Tuple):
        """Draw a filled rectangle."""
        pts = np.array([
            [x, y], [x + w, y], [x + w, y + h],
            [x, y], [x + w, y + h], [x, y + h],
        ], dtype=np.float32)
        self._use(self._prog)
        glUniform4f(self._loc(self._prog, "uColor"), *color)
        self._buf.upload(pts)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)

    def draw_rect_outline(self, x: float, y: float, w: float, h: float,
                          color: Tuple, lw: float = 1.0):
        """Draw a rectangle outline."""
        pts = np.array([
            [x, y], [x + w, y], [x + w, y], [x + w, y + h],
            [x + w, y + h], [x, y + h], [x, y + h], [x, y],
        ], dtype=np.float32)
        self.draw_lines(pts, color, lw)


class PlotCanvas(CanvasBase, QOpenGLWidget):
    def __init__(self, chart: "ChartWidget"):
        QOpenGLWidget.__init__(self, chart)
        self._chart = chart
        self._renderer: Optional[_GLRenderer] = None
        self._init_base_state()
        surf_fmt = QSurfaceFormat()
        surf_fmt.setSamples(8)
        surf_fmt.setSwapInterval(1)
        self.setFormat(surf_fmt)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def initializeGL(self):
        if self._renderer is not None:
            return
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

    def _show_context_menu(self, pos):
        from PyQt5.QtGui import QCursor
        from PyQt5.QtCore import QTimer
        menu = self._build_context_menu()
        gpos = QCursor.pos()
        QTimer.singleShot(0, lambda: menu.exec_(gpos))

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
        pw_f, ph_f = float(pr.width()), float(pr.height())
        pl_f, pb_f = float(pr.left()), float(pr.bottom())
        n_gl = len(xt_screen) + len(yt_screen)
        if n_gl:
            grid_arr = np.empty((n_gl * 2, 2), dtype=np.float32)
            idx = 0
            for tv in xt_screen:
                sx = pl_f + (tv - x0) / dx * pw_f
                grid_arr[idx] = [sx, pr.top()]
                grid_arr[idx + 1] = [sx, pr.bottom()]
                idx += 2
            for tv in yt_screen:
                sy = pb_f - (tv - y0) / dy * ph_f
                grid_arr[idx] = [pr.left(), sy]
                grid_arr[idx + 1] = [pr.right(), sy]
                idx += 2
            self._renderer.draw_lines(grid_arr, gr_col, 1.0)
        ax_col = (*fg_gl[:3], 0.32)
        self._renderer.draw_rect_outline(pr.left(), pr.top(), pr.width(), pr.height(), ax_col, 1.0)
        if self._range_sel_x is not None:
            rx_lo, rx_hi = sorted(self._range_sel_x)
            lx_lo = to_log(rx_lo) if log_x else rx_lo
            lx_hi = to_log(rx_hi) if log_x else rx_hi
            sx_lo = max(float(pr.left()), pl_f + (lx_lo - x0) / dx * pw_f)
            sx_hi = min(float(pr.right()), pl_f + (lx_hi - x0) / dx * pw_f)
            if sx_hi > sx_lo:
                sel_col = (*fg_gl[:3], _RANGE_SEL_ALPHA / 255.0)
                self._renderer.draw_rect_fill(sx_lo, pr.top(), sx_hi - sx_lo, pr.height(), sel_col)
                sel_border = (*fg_gl[:3], 0.4)
                self._renderer.draw_lines(np.array([
                    [sx_lo, pr.top()], [sx_lo, pr.bottom()],
                    [sx_hi, pr.top()], [sx_hi, pr.bottom()],
                ], dtype=np.float32), sel_border, 1.0)
        pl, pb, pw, ph = pl_f, pb_f, pw_f, ph_f
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
            self._renderer.draw_polyline(pts, fc, fit.pen.widthF() or 1.0, fit.pen.style())
        for fn_item in c.functions:
            if not fn_item.visible:
                continue
            fn_xs, fn_ys = fn_item.evaluate(x_lo_fn, x_hi_fn, max(1, pr.width()), max(1, pr.height()))
            if len(fn_xs) < 2:
                continue
            mask = np.array([y is not None for y in fn_ys])
            if mask.sum() < 2:
                continue
            vxs = [fn_xs[i] for i in range(len(fn_xs)) if mask[i]]
            vys = [fn_ys[i] for i in range(len(fn_ys)) if mask[i]]
            pts = _log_pts_to_screen(vxs, vys, x0, dx, y0, dy, pl, pb, pw, ph, log_x, log_y)
            fc = _qcolor_to_gl(fn_item.pen.color())
            self._renderer.draw_polyline(pts, fc, fn_item.pen.widthF() or 1.0, fn_item.pen.style())
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
                fill_col = (*lc[:3], item.fill_alpha / 255.0)
                self._renderer.draw_fill_under(pts, baseline_y, fill_col)
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
            dy_eff = ((to_log(c.vy1_r) - to_log(c.vy0_r)) if log_y else (c.vy1_r - c.vy0_r)) if is_r2 else dy
            if is_r2 and abs(dy_eff) < 1e-300:
                dy_eff = 1.0
            dxs, dys = decimated(item.xs, item.ys, _DECIMATE_THRESHOLD)
            pts = _log_pts_to_screen(dxs, dys, x0, dx, y0_eff, dy_eff, pl, pb, pw, ph, log_x, log_y)
            r = item.size / 2.0
            ic = _qcolor_to_gl(item.color)
            self._renderer.draw_circles_batch(pts, r, ic, sel_idx=item.selected_idx)
        glDisable(GL_SCISSOR_TEST)
        if self._mouse_pos is not None and not self._rb_active:
            self._paint_crosshair_gl(pr, x0, dx, y0, dy, fg_gl, bg_gl)
        if self._rb_active and self._rb_start and self._rb_end:
            rb = QRectF(self._rb_start, self._rb_end).normalized()
            self._renderer.draw_rect_fill(rb.left(), rb.top(), rb.width(), rb.height(), (*fg_gl[:3], 0.16))
            self._renderer.draw_rect_outline(rb.left(), rb.top(), rb.width(), rb.height(), (*fg_gl[:3], 0.63), 1.0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)
        self._renderer._last_prog = -1
        p.endNativePainting()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = QFontMetrics(c.font)
        p.setFont(c.font)
        lb_col = QColor(fg); lb_col.setAlpha(255)
        p.setPen(lb_col)
        for tv, ts in zip(xt, xt_screen):
            sx = int(pr.left() + (ts - x0) / dx * pr.width())
            lbl = _fmt_axis(tv)
            lw_txt = fm.horizontalAdvance(lbl)
            p.drawText(sx - lw_txt // 2, pr.bottom() + fm.height() + 2, lbl)
        for tv, ts in zip(yt, yt_screen):
            sy = int(pr.bottom() - (ts - y0) / dy * pr.height())
            lbl = _fmt_axis(tv)
            lw_txt = fm.horizontalAdvance(lbl)
            p.drawText(pr.left() - lw_txt - 6, sy + fm.ascent() // 2, lbl)
        if self._has_right_axis():
            y0r = to_log(c.vy0_r) if c.log_y else c.vy0_r
            y1r = to_log(c.vy1_r) if c.log_y else c.vy1_r
            dyr = y1r - y0r if abs(y1r - y0r) > 1e-300 else 1.0
            ytr = nice_log_ticks(10 ** y0r, 10 ** y1r) if log_y else nice_ticks(y0r, y1r, max(2, pr.height() // max(1, c.grid_px_y)))
            ytr_screen = [to_log(v) for v in ytr] if log_y else ytr
            r2_col = QColor(fg); r2_col.setAlpha(160)
            p.setPen(r2_col)
            for tv, ts in zip(ytr, ytr_screen):
                sy = int(pr.bottom() - (ts - y0r) / dyr * pr.height())
                lbl = _fmt_axis(tv)
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
                    self._draw_formula_tag(p, mid_pt, formula, fg, bg, fm)
        if self._mouse_pos is not None and not self._rb_active:
            self._paint_crosshair_overlay(p, pr, x0, dx, y0, dy, fm, fg, bg)
        if self._show_latest:
            self._paint_latest_points(p, pr, x0, dx, y0, dy, fm)
        if self._show_analytics:
            self._paint_analytics(p, pr, fg, bg, fm)
        if c.show_legend:
            self._paint_legend(p, pr, fg, bg, fm)
        for ruler in c.rulers:
            if ruler.visible:
                self._paint_ruler(p, ruler, pr, x0, dx, y0, dy, fg, bg, fm)
        p.end()

    def _paint_crosshair_gl(self, pr, x0, dx, y0, dy, fg_gl, bg_gl):
        """Draw GL crosshair lines and snap dot at nearest graph point."""
        mp = self._mouse_pos
        c = self._chart
        nearest = self._find_nearest(mp, pr, x0, dx, y0, dy)
        ch_col = (*fg_gl[:3], 0.31)
        if nearest is None:
            ch_pts = np.array([
                [mp.x(), pr.top()], [mp.x(), pr.bottom()],
                [pr.left(), mp.y()], [pr.right(), mp.y()],
            ], dtype=np.float32)
            self._renderer.draw_lines(ch_pts, ch_col, 1.0)
            return
        xi, yi, _, item = nearest
        snap = self._to_pt_log(xi, yi, x0, dx, y0, dy, pr, c.log_x, c.log_y)
        ch_pts = np.array([
            [snap.x(), pr.top()], [snap.x(), pr.bottom()],
            [pr.left(), snap.y()], [pr.right(), snap.y()],
        ], dtype=np.float32)
        self._renderer.draw_lines(ch_pts, ch_col, 1.0)
        self._renderer.draw_circle_fill(snap.x(), snap.y(), _SNAP_DOT_R, (*fg_gl[:3], 0.86))