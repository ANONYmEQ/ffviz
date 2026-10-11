"""Internal consistency checks between the USGS products and the parsed model.

Each check returns PASS / WARN / FAIL with a short explanation.  The checks
are printed on the console and summarised on the figure so that a reader can
see that the plotted model is the one USGS published.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .ffm import FiniteFaultModel, mw_from_m0
from .geometry import LocalFrame, segment_outline, plane_coords, distance_to_plane_km

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _rel(a, b):
    return abs(a - b) / abs(b) if b else float("inf")


def run_checks(model: FiniteFaultModel, frame: LocalFrame, ffm_props: dict, mt, event,
               catalog=None) -> List[Check]:
    out: List[Check] = []
    g = lambda k: _float(ffm_props.get(k))

    # 1. grid completeness
    bad = [s for s in model.segments if s.nx and s.ny and s.n != s.nx * s.ny]
    if any(s.nx for s in model.segments):
        out.append(Check("Subfault grid complete", FAIL if bad else PASS,
                         "; ".join(f"seg {s.index}: {s.n} vs nx*ny={s.nx * s.ny}" for s in bad)
                         or ", ".join(f"seg {s.index}: {s.nx}x{s.ny}" for s in model.segments)))

    # 2. grid regularity (spacing of centres equals Dx, Dy)
    msgs, worst = [], 0.0
    for s in model.segments:
        pc = plane_coords(s, frame)
        for vals, step, lab in ((pc.along, s.dx_km, "Dx"), (pc.downdip, s.dy_km, "Dy")):
            u = np.unique(np.round(vals, 1))
            if len(u) > 1 and step:
                d = np.diff(u)
                d = d[d > 0.3 * step]
                if len(d):
                    err = float(np.max(np.abs(d - step)) / step)
                    worst = max(worst, err)
                    if err > 0.05:
                        msgs.append(f"seg {s.index} {lab}: spacing deviates {100 * err:.0f}%")
    out.append(Check("Grid spacing = Dx, Dy", WARN if msgs else PASS,
                     "; ".join(msgs) or f"max deviation {100 * worst:.1f}%"))

    # 3. geometry reproduces the published segment outline
    # The header block may list either the outer edges of the grid or the
    # centres of the four corner subfaults; both are accepted.
    d_edge, d_cent = [], []
    for s in model.segments:
        if len(s.boundary) >= 4:
            ours = segment_outline(s, frame)
            pc = plane_coords(s, frame)
            cent = []
            for a, w in ((pc.along.min(), pc.downdip.min()), (pc.along.max(), pc.downdip.min()),
                         (pc.along.max(), pc.downdip.max()), (pc.along.min(), pc.downdip.max())):
                i = int(np.argmin((pc.along - a) ** 2 + (pc.downdip - w) ** 2))
                cent.append((s.lon[i], s.lat[i], s.depth[i]))
            for ref, acc in ((ours, d_edge), (cent, d_cent)):
                ox, oy = frame.to_xy([p[0] for p in ref], [p[1] for p in ref])
                oz = np.array([p[2] for p in ref])
                for lon, lat, dep in s.boundary[:4]:
                    bx, by = frame.to_xy(lon, lat)
                    acc.append(float(np.min(np.sqrt((ox - bx) ** 2 + (oy - by) ** 2 + (oz - dep) ** 2))))
    if d_edge:
        me, mc = max(d_edge), max(d_cent)
        mx, conv = (me, "grid edges") if me <= mc else (mc, "corner-subfault centres")
        out.append(Check("Outline vs published corners", PASS if mx < 1.5 else (WARN if mx < 5 else FAIL),
                         f"max mismatch {mx:.2f} km (header lists {conv})"))

    # 4. units / rigidity
    if model.rigidity_pa:
        r = model.rigidity_pa / 1e9
        out.append(Check("Implied rigidity", PASS if 10 <= r <= 80 else WARN,
                         f"{r:.0f} GPa (slip in {model.slip_unit}, moment in {model.moment_unit})"))

    # 5. total moment vs product
    m0_ffm = model.total_moment
    m0_ref = g("scalar-moment")
    if m0_ffm and m0_ref:
        e = _rel(m0_ffm, m0_ref)
        out.append(Check("Sum of subfault moments", PASS if e < 0.05 else (WARN if e < 0.2 else FAIL),
                         f"{m0_ffm:.3e} vs USGS {m0_ref:.3e} N-m ({100 * e:.1f}%)"))

    # 6. magnitude vs moment tensor
    if m0_ffm and mt is not None and mt.mw:
        d = mw_from_m0(m0_ffm) - mt.mw
        out.append(Check("Mw (FFM) vs Mw (moment tensor)", PASS if abs(d) <= 0.1 else WARN,
                         f"{mw_from_m0(m0_ffm):.2f} vs {mt.mw:.2f} ({d:+.2f})"))

    # 7. max slip vs product
    ms_ref = g("maximum-slip")
    if ms_ref:
        e = _rel(model.max_slip, ms_ref)
        out.append(Check("Maximum slip", PASS if e < 0.02 else (WARN if e < 0.1 else FAIL),
                         f"{model.max_slip:.2f} vs USGS {ms_ref:.2f} m"))

    # 8. hypocentre lies on the model
    if event is not None and not math.isnan(event.depth):
        d = min(distance_to_plane_km(s, frame, event.lon, event.lat, event.depth) for s in model.segments)
        out.append(Check("Hypocentre-to-fault distance", PASS if d < 10 else WARN,
                         f"{d:.1f} km from nearest segment plane (ComCat origin)"))

    # 9. mechanism consistency: model strike/dip vs a nodal plane
    if mt is not None and mt.np1 and mt.np2:
        main = max(model.segments, key=lambda s: np.nansum(s.moment) if s.moment is not None else s.n)
        best = min(_plane_angle(main.mean_strike, main.mean_dip, *p[:2]) for p in (mt.np1, mt.np2))
        out.append(Check("Fault plane vs MT nodal plane", PASS if best < 20 else WARN,
                         f"{best:.0f} deg between main segment and closest nodal plane"))

    # 10. catalogue information
    if catalog is not None:
        n, nf = len(catalog), int(np.sum(catalog.fixed_depth))
        out.append(Check("Aftershock catalogue", INFO,
                         f"{n} events; {nf} with default depth (excluded from section)"))
    for note in model.notes:
        out.append(Check("Note", INFO, note))
    return out


def _plane_angle(s1, d1, s2, d2):
    """Angle (deg) between two planes given by strike/dip (normal vectors)."""
    from .geometry import strike_dip_vectors

    def normal(s, d):
        sv, dv = strike_dip_vectors(np.array(s, float), np.array(d, float))
        return np.cross(sv, dv)
    c = abs(float(np.dot(normal(s1, d1), normal(s2, d2))))
    return math.degrees(math.acos(min(1.0, c)))


def _float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(checks: List[Check]) -> str:
    n = {k: sum(c.status == k for c in checks) for k in (PASS, WARN, FAIL)}
    return f"{n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail"
