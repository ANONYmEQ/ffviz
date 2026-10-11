"""Focal-mechanism utilities and beachball rendering (no ObsPy needed).

Conventions
-----------
* strike/dip/rake after Aki & Richards (1980).
* Moment tensors in the USE/(r, t, p) system used by GCMT and USGS:
  r = up, t = south, p = east.
* Beachballs: lower-hemisphere, equal-area (Schmidt) projection, north up;
  shaded quadrants = compressional P first motions (T-axis quadrants).
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def sdr_to_tensor(strike, dip, rake, m0=1.0):
    """Double-couple moment tensor (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)."""
    f, d, l = map(math.radians, (strike, dip, rake))
    sd, cd, s2d, c2d = math.sin(d), math.cos(d), math.sin(2 * d), math.cos(2 * d)
    sl, cl = math.sin(l), math.cos(l)
    sf, cf, s2f, c2f = math.sin(f), math.cos(f), math.sin(2 * f), math.cos(2 * f)
    mrr = m0 * s2d * sl
    mtt = -m0 * (sd * cl * s2f + s2d * sl * sf ** 2)
    mpp = m0 * (sd * cl * s2f - s2d * sl * cf ** 2)
    mrt = -m0 * (cd * cl * cf + c2d * sl * sf)
    mrp = m0 * (cd * cl * sf - c2d * sl * cf)
    mtp = -m0 * (sd * cl * c2f + 0.5 * s2d * sl * s2f)
    return mrr, mtt, mpp, mrt, mrp, mtp


def tensor_ned(m6: Sequence[float]) -> np.ndarray:
    """(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp) -> 3x3 tensor in (north, east, down)."""
    mrr, mtt, mpp, mrt, mrp, mtp = (float(v) for v in m6)
    return np.array([[mtt, -mtp, mrt],
                     [-mtp, mpp, -mrp],
                     [mrt, -mrp, mrr]])


def normal_slip(strike, dip, rake):
    """Fault normal n and slip vector u (hanging wall) in (north, east, down)."""
    f, d, l = map(math.radians, (strike, dip, rake))
    n = np.array([-math.sin(d) * math.sin(f), math.sin(d) * math.cos(f), -math.cos(d)])
    u = np.array([math.cos(l) * math.cos(f) + math.cos(d) * math.sin(l) * math.sin(f),
                  math.cos(l) * math.sin(f) - math.cos(d) * math.sin(l) * math.cos(f),
                  -math.sin(l) * math.sin(d)])
    return n, u


def _sdr_from_vectors(n, u) -> Tuple[float, float, float]:
    n, u = np.asarray(n, float), np.asarray(u, float)
    if n[2] > 0:                      # normal must point upward (negative "down")
        n, u = -n, -u
    dip = math.degrees(math.acos(max(-1.0, min(1.0, -n[2]))))
    sd = math.sin(math.radians(dip))
    if sd < 1e-9:                     # horizontal plane: strike arbitrary
        strike = 0.0
        rake = math.degrees(math.atan2(-u[1], u[0]))
        return strike, 0.0, rake
    strike = math.degrees(math.atan2(-n[0], n[1])) % 360.0
    f = math.radians(strike)
    sinl = -u[2] / sd
    cosl = u[0] * math.cos(f) + u[1] * math.sin(f)
    rake = math.degrees(math.atan2(sinl, cosl))
    return strike, dip, rake


def aux_plane(strike, dip, rake) -> Tuple[float, float, float]:
    """Strike/dip/rake of the auxiliary nodal plane."""
    n, u = normal_slip(strike, dip, rake)
    return _sdr_from_vectors(u, n)


def pt_axes(strike, dip, rake):
    """P and T axes as (trend, plunge) in degrees."""
    n, u = normal_slip(strike, dip, rake)
    out = []
    for v in ((n - u) / math.sqrt(2), (n + u) / math.sqrt(2)):
        if v[2] < 0:
            v = -v
        trend = math.degrees(math.atan2(v[1], v[0])) % 360.0
        plunge = math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))
        out.append((trend, plunge))
    return tuple(out)   # (P, T)


def project(trend_deg, plunge_deg):
    """Lower-hemisphere equal-area projection of a line -> (x=east, y=north) in the unit disk."""
    i = math.radians(90.0 - plunge_deg)            # angle from the downward vertical
    rho = math.sin(i / 2) / math.sin(math.pi / 4)
    t = math.radians(trend_deg)
    return rho * math.sin(t), rho * math.cos(t)


def radiation_grid(m6, n=301):
    """P-wave radiation amplitude on an (x, y) grid of the projection disk."""
    M = tensor_ned(m6)
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    rho = np.hypot(X, Y)
    inside = rho <= 1.0
    i = 2 * np.arcsin(np.clip(rho * math.sin(math.pi / 4), 0, 1))
    az = np.arctan2(X, Y)
    g = np.stack([np.sin(i) * np.cos(az), np.sin(i) * np.sin(az), np.cos(i)], axis=-1)
    A = np.einsum("...i,ij,...j->...", g, M, g)
    A[~inside] = np.nan
    return X, Y, A


def draw_beachball(ax, m6=None, sdr=None, color="#b3121a", edgecolor="#111111", lw=1.0, n=301):
    """Draw a beachball in a dedicated equal-aspect axes (unit circle)."""
    from matplotlib.patches import Circle
    if m6 is None:
        if sdr is None:
            raise ValueError("need a moment tensor or strike/dip/rake")
        m6 = sdr_to_tensor(*sdr)
    X, Y, A = radiation_grid(m6, n)
    scale = np.nanmax(np.abs(A)) or 1.0
    clip = Circle((0, 0), 1.0, transform=ax.transData)
    ax.add_patch(Circle((0, 0), 1.0, facecolor="white", edgecolor="none", zorder=1))
    cf = ax.contourf(X, Y, A / scale, levels=[0, 2], colors=[color], zorder=2)
    cl = ax.contour(X, Y, A / scale, levels=[0], colors=[edgecolor], linewidths=0.6 * lw, zorder=3)
    for coll in (cf, cl):
        try:
            coll.set_clip_path(clip)
        except AttributeError:          # matplotlib < 3.8
            for c in coll.collections:
                c.set_clip_path(clip)
    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor=edgecolor, linewidth=lw, zorder=4))
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.set_aspect("equal")
    ax.axis("off")
