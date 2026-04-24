"""
Automatic canvas backend selection.

Priority order:
  1. OpenGL (PyOpenGL + numpy available AND runtime GL context is usable)
  2. Software QPainter fallback (_PlotCanvas from canvas.py)

The GL probe creates a temporary QOffscreenSurface + QOpenGLContext to verify
that OpenGL is actually functional at runtime before committing to the GL
backend.  This prevents the Access Violation crash on Windows that occurs when
QOpenGLWidget is instantiated on systems where the GL driver is broken, missing,
or the surface cannot be created inside a QTabWidget before the window is shown.
"""
from __future__ import annotations

_GL_AVAILABLE = False
_GL_ERROR: str = ""
_PROBED = False


def _probe_gl() -> bool:
    global _GL_ERROR
    try:
        import numpy
        import OpenGL.GL
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
        fmt.setVersion(2, 1)
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
        from OpenGL.GL import glGetString, GL_VERSION
        ver = glGetString(GL_VERSION)
        ctx.doneCurrent()
        surface.destroy()
        if ver is None:
            _GL_ERROR = "glGetString(GL_VERSION) returned None"
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
    """
    Instantiate the best available canvas for *chart*.

    Returns a PlotCanvas (OpenGL) when OpenGL is confirmed functional at runtime,
    otherwise falls back to the software _PlotCanvas transparently.
    Both expose an identical public interface.
    """
    global _GL_ERROR
    _ensure_probed()
    if _GL_AVAILABLE:
        from .gl_canvas import PlotCanvas
        return PlotCanvas(chart)
    else:
        _GL_ERROR = "gl_canvas is not available"
        from .canvas import _PlotCanvas
        return _PlotCanvas(chart)


def backend_name() -> str:
    """Return 'opengl' or 'software'."""
    _ensure_probed()
    return "opengl" if _GL_AVAILABLE else "software"


def gl_available() -> bool:
    """Return True if the OpenGL backend is available and will be used."""
    _ensure_probed()
    return _GL_AVAILABLE


def gl_error() -> str:
    """Return the import error string when OpenGL is unavailable, else ''."""
    _ensure_probed()
    return _GL_ERROR