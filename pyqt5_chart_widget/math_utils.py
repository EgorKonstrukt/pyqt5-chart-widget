from __future__ import annotations
import math
from bisect import bisect_right
from typing import Callable, Dict, List, Optional, Tuple


def linspace(start: float, stop: float, n: int) -> List[float]:
    if n < 2:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + step * i for i in range(n)]


def _sort_unique(x: List[float], y: List[float]) -> Tuple[List[float], List[float]]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    x2, y2 = [x[i] for i in order], [y[i] for i in order]
    xs: List[float] = []
    ys: List[float] = []
    for xi, yi in zip(x2, y2):
        if not xs or xi != xs[-1]:
            xs.append(xi)
            ys.append(yi)
    return xs, ys


def _gauss_solve(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        if abs(a[col][col]) < 1e-15:
            continue
        f = a[col][col]
        a[col] = [v / f for v in a[col]]
        b[col] /= f
        for row in range(n):
            if row == col:
                continue
            fac = a[row][col]
            a[row] = [a[row][j] - fac * a[col][j] for j in range(n)]
            b[row] -= fac * b[col]
    return b


def _polyfit(x: List[float], y: List[float], deg: int) -> List[float]:
    d = deg + 1
    vt_v = [[0.0] * d for _ in range(d)]
    vt_y = [0.0] * d
    for xi, yi in zip(x, y):
        row = [xi ** (deg - j) for j in range(d)]
        for r in range(d):
            for c in range(d):
                vt_v[r][c] += row[r] * row[c]
            vt_y[r] += row[r] * yi
    return _gauss_solve(vt_v, vt_y)


def _polyval(coeffs: List[float], x: float) -> float:
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


def _pchip_eval(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> List[float]:
    n = len(x_pts)
    h = [x_pts[i + 1] - x_pts[i] for i in range(n - 1)]
    s = [(y_pts[i + 1] - y_pts[i]) / h[i] for i in range(n - 1)]
    d = [0.0] * n
    d[0] = s[0]
    d[-1] = s[-1]
    for k in range(1, n - 1):
        if s[k - 1] * s[k] <= 0.0:
            d[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k])
    result = []
    for xi in x_eval:
        idx = max(0, min(bisect_right(x_pts, xi) - 1, n - 2))
        dx = xi - x_pts[idx]
        hk = h[idx]
        t = dx / hk
        t2 = t * t
        t3 = t2 * t
        result.append(
            (2 * t3 - 3 * t2 + 1) * y_pts[idx]
            + (t3 - 2 * t2 + t) * hk * d[idx]
            + (-2 * t3 + 3 * t2) * y_pts[idx + 1]
            + (t3 - t2) * hk * d[idx + 1]
        )
    return result


def _cubic_spline_eval(x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> List[float]:
    n = len(x_pts)
    h = [float(x_pts[i + 1] - x_pts[i]) for i in range(n - 1)]
    rhs = [0.0] * n
    for i in range(1, n - 1):
        rhs[i] = 3.0 * ((y_pts[i + 1] - y_pts[i]) / h[i] - (y_pts[i] - y_pts[i - 1]) / h[i - 1])
    diag = [2.0] * n
    lo = [0.0] * n
    up = [0.0] * n
    for i in range(1, n - 1):
        lo[i] = h[i - 1]
        up[i] = h[i]
    diag[0] = 1.0; up[0] = 0.0; rhs[0] = 0.0
    diag[-1] = 1.0; lo[-1] = 0.0; rhs[-1] = 0.0
    cu = [0.0] * n
    cr = [0.0] * n
    cu[0] = up[0] / diag[0] if diag[0] else 0.0
    cr[0] = rhs[0] / diag[0] if diag[0] else 0.0
    for i in range(1, n):
        den = diag[i] - lo[i] * cu[i - 1]
        cu[i] = (up[i] / den) if (i < n - 1 and den) else 0.0
        cr[i] = ((rhs[i] - lo[i] * cr[i - 1]) / den) if den else 0.0
    m = [0.0] * n
    m[-1] = cr[-1]
    for i in range(n - 2, -1, -1):
        m[i] = cr[i] - cu[i] * m[i + 1]
    a = list(y_pts[:-1])
    b_c = [(y_pts[i + 1] - y_pts[i]) / h[i] - h[i] * (2 * m[i] + m[i + 1]) / 3.0 for i in range(n - 1)]
    c_c = m[:-1]
    d_c = [(m[i + 1] - m[i]) / (3.0 * h[i]) for i in range(n - 1)]
    result = []
    for xi in x_eval:
        idx = max(0, min(bisect_right(x_pts, xi) - 1, n - 2))
        dx = xi - x_pts[idx]
        result.append(a[idx] + b_c[idx] * dx + c_c[idx] * dx ** 2 + d_c[idx] * dx ** 3)
    return result


FitFn = Callable[[List[float], List[float], List[float]], List[float]]


class FitMode:
    def __init__(self, key: str, label: str, fn: FitFn, min_points: int = 2):
        self.key = key
        self.label = label
        self.fn = fn
        self.min_points = min_points

    def evaluate(self, x_pts: List[float], y_pts: List[float], x_eval: List[float]) -> Optional[List[float]]:
        x_pts, y_pts = _sort_unique(x_pts, y_pts)
        if len(x_pts) < self.min_points:
            return None
        return self.fn(x_pts, y_pts, x_eval)


def _fit_linear_origin(x_pts, y_pts, x_eval):
    denom = sum(xi * xi for xi in x_pts)
    k = sum(xi * yi for xi, yi in zip(x_pts, y_pts)) / denom if denom else 0.0
    return [k * xi for xi in x_eval]


def _fit_linear(x_pts, y_pts, x_eval):
    x_min = x_pts[0]
    x_range = float(x_pts[-1] - x_min) or 1.0
    xn = [(xi - x_min) / x_range for xi in x_pts]
    c = _polyfit(xn, y_pts, 1)
    return [_polyval(c, (xi - x_min) / x_range) for xi in x_eval]


def _make_poly_fit(deg: int) -> FitFn:
    def _fit(x_pts, y_pts, x_eval):
        actual_deg = min(deg, len(x_pts) - 1)
        x_min = x_pts[0]
        x_range = float(x_pts[-1] - x_min) or 1.0
        xn = [(xi - x_min) / x_range for xi in x_pts]
        c = _polyfit(xn, y_pts, actual_deg)
        return [_polyval(c, (xi - x_min) / x_range) for xi in x_eval]
    return _fit


def _fit_pchip(x_pts, y_pts, x_eval):
    return _pchip_eval(x_pts, y_pts, x_eval)


def _fit_cubic_spline(x_pts, y_pts, x_eval):
    if len(x_pts) == 2:
        return _pchip_eval(x_pts, y_pts, x_eval)
    return _cubic_spline_eval(x_pts, y_pts, x_eval)


_BUILTIN_MODES: List[FitMode] = [
    FitMode("linear_origin", "Linear (origin)", _fit_linear_origin, 1),
    FitMode("linear",        "Linear",           _fit_linear,         2),
    FitMode("poly2",         "Polynomial 2°",    _make_poly_fit(2),   2),
    FitMode("poly3",         "Polynomial 3°",    _make_poly_fit(3),   2),
    FitMode("poly4",         "Polynomial 4°",    _make_poly_fit(4),   2),
    FitMode("pchip",         "PCHIP",            _fit_pchip,          2),
    FitMode("spline",        "Cubic Spline",      _fit_cubic_spline,   2),
]

_REGISTRY: Dict[str, FitMode] = {m.key: m for m in _BUILTIN_MODES}


def register_fit_mode(mode: FitMode):
    _REGISTRY[mode.key] = mode


def get_fit_modes() -> List[FitMode]:
    return list(_REGISTRY.values())


def get_fit_mode(key: str) -> Optional[FitMode]:
    return _REGISTRY.get(key)


def nice_ticks(lo: float, hi: float, n: int = 7) -> List[float]:
    if hi <= lo:
        return [lo]
    span = hi - lo
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
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


def fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1000 or (abs(v) < 0.001 and v != 0):
        return f"{v:.3g}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3g}"
