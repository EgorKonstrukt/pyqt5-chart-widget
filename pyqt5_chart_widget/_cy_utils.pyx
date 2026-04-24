# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport hypot, log10, fabs
from libc.float cimport DBL_MAX
from libc.stdlib cimport malloc, free
import array as _am
from cpython.array cimport array, clone

cdef array _DBL_T = _am.array('d')

DEF MAX_D = 10
DEF MAX_D2 = 100
DEF SENTINEL = 1e308
DEF DP_EPS_SQ = 0.25


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


cdef int _dp_simplify(
    double* px, double* py, int n,
    double eps_sq,
    double* ox, double* oy
) noexcept nogil:
    cdef int i, j, lo, hi, mid, sp, out_n
    cdef double ax, ay, bx, by, abx, aby, ab2, t, nx, ny, dx2, dy2, d2, max_d2
    cdef int* stack
    cdef bint* keep
    if n <= 2:
        for i in range(n):
            ox[i] = px[i]; oy[i] = py[i]
        return n
    stack = <int*>malloc(n * 2 * sizeof(int))
    keep = <bint*>malloc(n * sizeof(bint))
    sp = 0; out_n = 0
    for i in range(n):
        keep[i] = 0
    keep[0] = 1; keep[n - 1] = 1
    stack[sp] = 0; sp += 1
    stack[sp] = n - 1; sp += 1
    while sp >= 2:
        sp -= 1; hi = stack[sp]
        sp -= 1; lo = stack[sp]
        if hi - lo <= 1:
            continue
        ax = px[lo]; ay = py[lo]
        bx = px[hi]; by = py[hi]
        abx = bx - ax; aby = by - ay
        ab2 = abx * abx + aby * aby
        max_d2 = -1.0; mid = lo + 1
        for i in range(lo + 1, hi):
            if ab2 < 1e-15:
                dx2 = px[i] - ax; dy2 = py[i] - ay
                d2 = dx2 * dx2 + dy2 * dy2
            else:
                t = ((px[i] - ax) * abx + (py[i] - ay) * aby) / ab2
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                nx = ax + t * abx - px[i]
                ny = ay + t * aby - py[i]
                d2 = nx * nx + ny * ny
            if d2 > max_d2:
                max_d2 = d2; mid = i
        if max_d2 > eps_sq:
            keep[mid] = 1
            stack[sp] = lo; sp += 1
            stack[sp] = mid; sp += 1
            stack[sp] = mid; sp += 1
            stack[sp] = hi; sp += 1
    for i in range(n):
        if keep[i]:
            ox[out_n] = px[i]; oy[out_n] = py[i]
            out_n += 1
    free(stack); free(keep)
    return out_n


def nearest_on_segments_cy(
    double mx, double my,
    list xs, list ys,
    double x0, double dx, double y0, double dy,
    double pl, double pb, double pw, double ph,
    bint lx, bint ly,
):
    cdef int n, i
    cdef double bd, bxi, byi
    cdef double p0x, p0y, p1x, p1y, sx, sy, sl2, t, d
    cdef double xi0, yi0, xi1, yi1
    n = len(xs)
    if n == 0:
        return None
    bd = DBL_MAX; bxi = 0.0; byi = 0.0
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
    cdef int n, i
    cdef double lo, hi, total, px, py, cx, cy
    cdef bint hp
    n = len(xs)
    if n < 2:
        return 0.0
    lo = <double>x_lo if x_lo is not None else -DBL_MAX
    hi = <double>x_hi if x_hi is not None else DBL_MAX
    total = 0.0; px = 0.0; py = 0.0; hp = False
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
    cdef int n, i, j, seg_start, seg_len, simp_n, out_n
    cdef double px2, py2
    cdef double* raw_px
    cdef double* raw_py
    cdef double* simp_px
    cdef double* simp_py
    cdef array empty, out_buf
    cdef double[:] v
    cdef object yi_obj
    n = len(xs)
    if n == 0:
        empty = clone(_DBL_T, 0, False)
        return empty, 0
    raw_px = <double*>malloc(n * sizeof(double))
    raw_py = <double*>malloc(n * sizeof(double))
    simp_px = <double*>malloc(n * sizeof(double))
    simp_py = <double*>malloc(n * sizeof(double))
    out_buf = clone(_DBL_T, n * 2, False)
    v = out_buf
    out_n = 0
    seg_start = -1
    for i in range(n):
        yi_obj = ys[i]
        if yi_obj is None:
            if seg_start >= 0:
                seg_len = i - seg_start
                if seg_len >= 2:
                    simp_n = _dp_simplify(
                        raw_px + seg_start, raw_py + seg_start, seg_len,
                        DP_EPS_SQ, simp_px, simp_py
                    )
                    if out_n > 0:
                        v[out_n * 2] = SENTINEL; v[out_n * 2 + 1] = SENTINEL
                        out_n += 1
                    for j in range(simp_n):
                        v[out_n * 2] = simp_px[j]; v[out_n * 2 + 1] = simp_py[j]
                        out_n += 1
                elif seg_len == 1:
                    if out_n > 0:
                        v[out_n * 2] = SENTINEL; v[out_n * 2 + 1] = SENTINEL
                        out_n += 1
                    v[out_n * 2] = raw_px[seg_start]; v[out_n * 2 + 1] = raw_py[seg_start]
                    out_n += 1
            seg_start = -1
        else:
            px2 = _cl(pl + ((<double>xs[i] - x0) / dx * pw))
            py2 = _cl(pb - ((<double>yi_obj - y0) / dy * ph))
            if seg_start < 0:
                seg_start = i
            raw_px[i] = px2
            raw_py[i] = py2
    if seg_start >= 0:
        seg_len = n - seg_start
        if seg_len >= 2:
            simp_n = _dp_simplify(
                raw_px + seg_start, raw_py + seg_start, seg_len,
                DP_EPS_SQ, simp_px, simp_py
            )
            if out_n > 0:
                v[out_n * 2] = SENTINEL; v[out_n * 2 + 1] = SENTINEL
                out_n += 1
            for j in range(simp_n):
                v[out_n * 2] = simp_px[j]; v[out_n * 2 + 1] = simp_py[j]
                out_n += 1
        elif seg_len == 1:
            if out_n > 0:
                v[out_n * 2] = SENTINEL; v[out_n * 2 + 1] = SENTINEL
                out_n += 1
            v[out_n * 2] = raw_px[seg_start]; v[out_n * 2 + 1] = raw_py[seg_start]
            out_n += 1
    free(raw_px); free(raw_py); free(simp_px); free(simp_py)
    return out_buf, out_n


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
