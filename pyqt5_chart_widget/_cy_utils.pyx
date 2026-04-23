# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport hypot, log10, fabs
from libc.float cimport DBL_MAX
import array as _am
from cpython.array cimport array, clone

cdef array _DBL_T = _am.array('d')

DEF MAX_D = 10
DEF MAX_D2 = 100


cdef inline double _cl(double v) noexcept nogil:
    if v > 1e6: return 1e6
    if v < -1e6: return -1e6
    return v


cdef inline double _lg(double v) noexcept nogil:
    return log10(v) if v > 0.0 else -1e308


cdef inline void _sc(
    double xv, double yv,
    double x0, double dx, double y0, double dy,
    double pl, double pb, double pw, double ph,
    bint lx, bint ly,
    double* ox, double* oy
) noexcept nogil:
    ox[0] = _cl(pl + ((_lg(xv) if lx else xv) - x0) / dx * pw)
    oy[0] = _cl(pb - ((_lg(yv) if ly else yv) - y0) / dy * ph)


def nearest_on_segments_cy(
    double mx, double my,
    list xs, list ys,
    double x0, double dx, double y0, double dy,
    double pl, double pb, double pw, double ph,
    bint lx, bint ly,
):
    cdef int n = len(xs)
    if n == 0:
        return None
    cdef double bd = DBL_MAX, bxi = 0.0, byi = 0.0
    cdef double p0x, p0y, p1x, p1y, sx, sy, sl2, t, d
    cdef double xi0, yi0, xi1, yi1
    cdef int i
    xi0 = <double>xs[0]; yi0 = <double>ys[0]
    _sc(xi0, yi0, x0, dx, y0, dy, pl, pb, pw, ph, lx, ly, &p0x, &p0y)
    if n == 1:
        return xi0, yi0, hypot(p0x - mx, p0y - my)
    for i in range(n - 1):
        xi1 = <double>xs[i + 1]; yi1 = <double>ys[i + 1]
        _sc(xi1, yi1, x0, dx, y0, dy, pl, pb, pw, ph, lx, ly, &p1x, &p1y)
        sx = p1x - p0x; sy = p1y - p0y
        sl2 = sx * sx + sy * sy
        if sl2 < 1e-10:
            t = 0.0
        else:
            t = ((mx - p0x) * sx + (my - p0y) * sy) / sl2
            if t < 0.0: t = 0.0
            elif t > 1.0: t = 1.0
        d = hypot(p0x + t * sx - mx, p0y + t * sy - my)
        if d < bd:
            bd = d; bxi = xi0 + t * (xi1 - xi0); byi = yi0 + t * (yi1 - yi0)
        p0x = p1x; p0y = p1y; xi0 = xi1; yi0 = yi1
    return bxi, byi, bd


def decimated_to_screen_cy(
    list xs, list ys, int max_pts,
    double x0, double dx, double y0, double dy,
    double pl, double pb, double pw, double ph,
    bint lx, bint ly,
):
    cdef int n = len(xs)
    cdef int out_n = n if n <= max_pts else max_pts
    cdef array buf = clone(_DBL_T, out_n * 2, False)
    cdef double[:] v = buf
    cdef double step = (<double>n / out_n) if out_n < n else 1.0
    cdef double px, py
    cdef int i, idx
    for i in range(out_n - 1):
        idx = <int>(i * step)
        _sc(<double>xs[idx], <double>ys[idx], x0, dx, y0, dy, pl, pb, pw, ph, lx, ly, &px, &py)
        v[i * 2] = px; v[i * 2 + 1] = py
    _sc(<double>xs[n - 1], <double>ys[n - 1], x0, dx, y0, dy, pl, pb, pw, ph, lx, ly, &px, &py)
    v[(out_n - 1) * 2] = px; v[(out_n - 1) * 2 + 1] = py
    return buf, out_n


def trapezoid_integral_cy(list xs, list ys, object x_lo, object x_hi):
    cdef int n = len(xs)
    if n < 2:
        return 0.0
    cdef double lo = <double>x_lo if x_lo is not None else -DBL_MAX
    cdef double hi = <double>x_hi if x_hi is not None else DBL_MAX
    cdef double total = 0.0, px = 0.0, py = 0.0, cx, cy
    cdef bint hp = False
    cdef int i
    for i in range(n):
        cx = <double>xs[i]; cy = <double>ys[i]
        if cx < lo or cx > hi:
            hp = False
            continue
        if hp:
            total += (cx - px) * (py + cy) * 0.5
        px = cx; py = cy; hp = True
    return total


cdef void _gauss_ip(double* a, double* b, int nd) noexcept nogil:
    cdef int col, row, pr, r
    cdef double f, tmp, av
    for col in range(nd):
        pr = col; av = fabs(a[col * nd + col])
        for r in range(col + 1, nd):
            if fabs(a[r * nd + col]) > av:
                av = fabs(a[r * nd + col]); pr = r
        if pr != col:
            for r in range(nd):
                tmp = a[col * nd + r]; a[col * nd + r] = a[pr * nd + r]; a[pr * nd + r] = tmp
            tmp = b[col]; b[col] = b[pr]; b[pr] = tmp
        if a[col * nd + col] == 0.0:
            continue
        f = a[col * nd + col]
        for r in range(nd):
            a[col * nd + r] /= f
        b[col] /= f
        for row in range(nd):
            if row == col: continue
            f = a[row * nd + col]
            for r in range(nd):
                a[row * nd + r] -= f * a[col * nd + r]
            b[row] -= f * b[col]


def fn_to_screen_cy(
    list xs, list ys,
    double x0, double dx, double y0, double dy,
    double pl, double pb, double pw, double ph,
):
    cdef int n = len(xs)
    cdef array buf = clone(_DBL_T, n * 2, False)
    cdef double[:] v = buf
    cdef double px, py
    cdef object yi_obj
    cdef int i
    for i in range(n):
        yi_obj = ys[i]
        if yi_obj is None:
            v[i * 2] = 1e308
            v[i * 2 + 1] = 1e308
        else:
            px = _cl(pl + ((<double>xs[i] - x0) / dx * pw))
            py = _cl(pb - ((<double>yi_obj - y0) / dy * ph))
            v[i * 2] = px
            v[i * 2 + 1] = py
    return buf, n


def polyfit_cy(list xn, list yn, int deg):
    cdef int d = deg + 1 if deg + 1 <= MAX_D else MAX_D
    cdef int n = len(xn), i, r, c, j
    cdef double vtv[MAX_D2]
    cdef double vty[MAX_D]
    cdef double row[MAX_D]
    cdef double xi, yi
    for r in range(d):
        vty[r] = 0.0
        for c in range(d):
            vtv[r * d + c] = 0.0
    for i in range(n):
        xi = <double>xn[i]; yi = <double>yn[i]
        row[d - 1] = 1.0
        for j in range(d - 2, -1, -1):
            row[j] = row[j + 1] * xi
        for r in range(d):
            for c in range(d):
                vtv[r * d + c] += row[r] * row[c]
            vty[r] += row[r] * yi
    _gauss_ip(vtv, vty, d)
    return [vty[i] for i in range(d)]
