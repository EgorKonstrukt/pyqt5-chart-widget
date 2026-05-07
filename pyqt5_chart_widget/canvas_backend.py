"""Automatic canvas backend selection.

Priority order:
  1. ModernGL (moderngl available AND runtime GL 3.3 context is usable)
  2. Software QPainter fallback (_PlotCanvas from canvas.py)

The GL probe creates a temporary QOffscreenSurface + QOpenGLContext to verify
that OpenGL 3.3 Core is functional before committing to the ModernGL backend.
"""
from __future__ import annotations

_GL_AVAILABLE = False
_GL_ERROR: str = ""
_PROBED = False


def _probe_gl() -> bool:
    global _GL_ERROR
    try:
        import moderngl
    except Exception as exc:
        _GL_ERROR = f"import failed: {exc}"
        return False
    try:
        from PyQt5.QtGui import QOpenGLContext, QSurfaceFormat, QOffscreenSurface
        from PyQt5.QtCore import QCoreApplication
        if QCoreApplication.instance() is None:
            _GL_ERROR = "no QApplication instance"
            return False
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        surface = QOffscreenSurface()
        surface.setFormat(fmt)
        surface.create()
        if not surface.isValid():
            _GL_ERROR = "QOffscreenSurface.create() failed"
            return False
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            _GL_ERROR = "QOpenGLContext.create() failed"
            return False
        if not ctx.makeCurrent(surface):
            _GL_ERROR = "makeCurrent() failed"
            return False
        try:
            mgl = moderngl.create_context()
            ver = mgl.version_code
            mgl.release()
        except Exception as exc:
            _GL_ERROR = f"moderngl context failed: {exc}"
            ctx.doneCurrent()
            surface.destroy()
            return False
        ctx.doneCurrent()
        surface.destroy()
        if ver < 330:
            _GL_ERROR = f"GL version {ver} < 330 required"
            return False
        return True
    except Exception as exc:
        _GL_ERROR = f"runtime probe failed: {exc}"
        return False


def _ensure_probed():
    global _GL_AVAILABLE, _PROBED
    if _PROBED:
        return
    _PROBED = True
    _GL_AVAILABLE = _probe_gl()


def make_canvas(chart) -> object:
    """Return best available canvas for *chart*: ModernGL or software fallback."""
    global _GL_ERROR
    _ensure_probed()
    if _GL_AVAILABLE:
        from .gl_canvas import PlotCanvas
        return PlotCanvas(chart)
    _GL_ERROR = "gl_canvas is not available"
    from .canvas import _PlotCanvas
    return _PlotCanvas(chart)


def backend_name() -> str:
    """Return 'opengl' or 'software'."""
    _ensure_probed()
    return "opengl" if _GL_AVAILABLE else "software"


def gl_available() -> bool:
    """Return True if the ModernGL backend is available and will be used."""
    _ensure_probed()
    return _GL_AVAILABLE


def gl_error() -> str:
    """Return the error string when OpenGL is unavailable, else ''."""
    _ensure_probed()
    return _GL_ERROR