"""Readers for USGS finite-fault model files.

Supported formats
-----------------
* ``*.param`` ("basic_inversion.param"), the native output of the
  Ji et al. (2002) inversion used by the USGS NEIC.  Per segment it contains
  a header with the grid dimensions (nx, ny, Dx, Dy), the segment outline
  (lon, lat, depth of the corners) and one line per subfault with
  lat, lon, depth, slip, rake, strike, dip, t_rup, t_ris, t_fal, mo.
* ``*.fsp`` (SRCMOD / "complete_inversion.fsp") as a fallback.

Units are *not* assumed: slip (cm or m) and seismic moment (dyne-cm or N-m)
are resolved by :func:`resolve_units` from the implied rigidity and, when
available, from the scalar moment / maximum slip reported in the USGS
product properties.  All quantities are stored internally in SI units
(slip in m, moment in N-m, lengths in km, times in s).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class FFMFormatError(ValueError):
    """Raised when a finite-fault file cannot be interpreted."""


@dataclass
class Segment:
    index: int
    nx: Optional[int] = None
    ny: Optional[int] = None
    dx_km: Optional[float] = None          # subfault length along strike
    dy_km: Optional[float] = None          # subfault width down dip
    boundary: List[tuple] = field(default_factory=list)   # (lon, lat, depth_km)
    hypo_cell: Optional[tuple] = None      # (i, j) as given in the header
    hypo_lonlat: Optional[tuple] = None    # (lon, lat) as given in the header
    seg_strike: Optional[float] = None     # FSP segment header values
    seg_dip: Optional[float] = None
    # per-subfault arrays (filled after parsing)
    lat: np.ndarray = None
    lon: np.ndarray = None
    depth: np.ndarray = None               # km, subfault centre
    slip_raw: np.ndarray = None            # as read from file
    rake: np.ndarray = None
    strike: np.ndarray = None
    dip: np.ndarray = None
    t_rup: Optional[np.ndarray] = None     # s
    t_rise: Optional[np.ndarray] = None    # s
    t_fall: Optional[np.ndarray] = None    # s
    moment_raw: Optional[np.ndarray] = None
    # resolved SI values
    slip: np.ndarray = None                # m
    moment: Optional[np.ndarray] = None    # N-m

    @property
    def n(self) -> int:
        return 0 if self.lat is None else len(self.lat)

    @property
    def mean_strike(self) -> float:
        """Circular mean of the subfault strikes (degrees, 0-360)."""
        s = np.radians(self.strike)
        return float(np.degrees(np.arctan2(np.sin(s).mean(), np.cos(s).mean())) % 360.0)

    @property
    def mean_dip(self) -> float:
        return float(np.mean(self.dip))


@dataclass
class FiniteFaultModel:
    segments: List[Segment]
    source_format: str                     # "param" or "fsp"
    slip_unit: str = "unknown"             # unit found in the file
    moment_unit: str = "unknown"
    rigidity_pa: Optional[float] = None    # median implied rigidity
    notes: List[str] = field(default_factory=list)

    # ---- convenience accessors -------------------------------------------------
    def _cat(self, name):
        arrs = [getattr(s, name) for s in self.segments]
        if any(a is None for a in arrs):
            return None
        return np.concatenate(arrs)

    @property
    def slip(self):
        return self._cat("slip")

    @property
    def moment(self):
        return self._cat("moment")

    @property
    def total_moment(self) -> Optional[float]:
        m = self.moment
        return None if m is None else float(np.nansum(m))

    @property
    def max_slip(self) -> float:
        return float(np.nanmax(self.slip))

    def subfault_area_m2(self, seg: Segment) -> float:
        if seg.dx_km is None or seg.dy_km is None:
            raise FFMFormatError(f"segment {seg.index}: subfault size unknown")
        return seg.dx_km * seg.dy_km * 1e6


# --------------------------------------------------------------------------------
# .param reader
# --------------------------------------------------------------------------------
_SEG_RE = re.compile(
    r"Fault_?segment\s*=\s*(\d+).*?nx\S*\s*=\s*(\d+).*?Dx\s*=\s*(" + _FLOAT + r")\s*km"
    r".*?ny\S*\s*=\s*(\d+).*?Dy\s*=\s*(" + _FLOAT + r")\s*km", re.I)
_BOUND_RE = re.compile(r"Boundary\s+of\s+Fault_?segment\s*(\d+)", re.I)
_CELL_RE = re.compile(r"cell\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.I)
_LONLAT_RE = re.compile(r"Lon\.?\s*:?\s*(" + _FLOAT + r").*?Lat\.?\s*:?\s*(" + _FLOAT + r")", re.I)

_PARAM_ALIASES = {
    "lat": "lat", "lon": "lon", "depth": "depth", "dep": "depth", "slip": "slip",
    "rake": "rake", "strike": "strike", "dip": "dip", "t_rup": "t_rup", "trup": "t_rup",
    "t_ris": "t_rise", "t_rise": "t_rise", "t_fal": "t_fall", "t_fall": "t_fall",
    "mo": "mo", "moment": "mo",
}


def _numbers(line: str) -> Optional[List[float]]:
    parts = line.split()
    if not parts:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def _norm_col(tok: str) -> str:
    return tok.strip().lower().rstrip(".").replace("-", "_")


def read_param(text: str) -> FiniteFaultModel:
    segments: Dict[int, Segment] = {}
    order: List[int] = []
    current: Optional[Segment] = None
    columns: Optional[List[str]] = None
    mode = None            # "boundary" | "data"
    rows: Dict[int, List[List[float]]] = {}
    skipped: List[str] = []

    def seg(idx: int) -> Segment:
        if idx not in segments:
            segments[idx] = Segment(index=idx)
            order.append(idx)
            rows[idx] = []
        return segments[idx]

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            m = _SEG_RE.search(body)
            if m:
                current = seg(int(m.group(1)))
                current.nx, current.dx_km = int(m.group(2)), float(m.group(3))
                current.ny, current.dy_km = int(m.group(4)), float(m.group(5))
                mode = None
                continue
            m = _BOUND_RE.search(body)
            if m:
                current = seg(int(m.group(1)))
                c = _CELL_RE.search(body)
                if c:
                    current.hypo_cell = (int(c.group(1)), int(c.group(2)))
                ll = _LONLAT_RE.search(body[c.end():] if c else body)
                if ll:
                    current.hypo_lonlat = (float(ll.group(1)), float(ll.group(2)))
                mode = "boundary"
                continue
            toks = [_norm_col(t) for t in body.split()]
            if "slip" in toks and ("lat" in toks or "lon" in toks):
                columns = [_PARAM_ALIASES.get(t, t) for t in toks]
                mode = "data"
                continue
            # any other comment line (e.g. "#Lon. Lat. Depth") keeps the mode
            continue

        nums = _numbers(line)
        if nums is None:
            continue
        if current is None:
            current = seg(1)
        if mode == "boundary" and len(nums) == 3:
            current.boundary.append(tuple(nums))
        elif mode == "data" and columns and len(nums) == len(columns):
            rows[current.index].append(nums)
        elif mode == "data" and columns:
            skipped.append(line)

    if not order or columns is None:
        raise FFMFormatError("no subfault table found (is this a USGS .param file?)")

    required = {"lat", "lon", "depth", "slip", "rake", "strike", "dip"}
    missing = required - set(columns)
    if missing:
        raise FFMFormatError(f"subfault table lacks columns: {sorted(missing)}")

    out = []
    for idx in order:
        s = segments[idx]
        data = np.array(rows[idx], dtype=float)
        if data.size == 0:
            continue
        col = {name: data[:, i] for i, name in enumerate(columns)}
        s.lat, s.lon, s.depth = col["lat"], col["lon"], col["depth"]
        s.slip_raw, s.rake = col["slip"], col["rake"]
        s.strike, s.dip = col["strike"] % 360.0, col["dip"]
        s.t_rup, s.t_rise, s.t_fall = col.get("t_rup"), col.get("t_rise"), col.get("t_fall")
        s.moment_raw = col.get("mo")
        out.append(s)
    model = FiniteFaultModel(segments=out, source_format="param")
    if skipped:
        model.notes.append(f"{len(skipped)} numeric line(s) with unexpected column count ignored, e.g. {skipped[0]!r}")
    for s in out:
        if s.nx and s.ny and s.n != s.nx * s.ny:
            model.notes.append(f"segment {s.index}: {s.n} subfaults but header says nx*ny={s.nx * s.ny}")
    return model


# --------------------------------------------------------------------------------
# .fsp reader (fallback)
# --------------------------------------------------------------------------------
_FSP_SEG_RE = re.compile(r"SEGMENT\s*#?\s*(\d+).*?STRIKE\s*=\s*(" + _FLOAT + r").*?DIP\s*=\s*(" + _FLOAT + r")", re.I)
_FSP_MECH_RE = re.compile(r"STRK\s*=\s*(" + _FLOAT + r").*?DIP\s*=\s*(" + _FLOAT + r")", re.I)
_FSP_DXDZ_RE = re.compile(r"Dx\s*=\s*(" + _FLOAT + r")\s*km.*?Dz\s*=\s*(" + _FLOAT + r")\s*km", re.I)
_FSP_NXNZ_RE = re.compile(r"Nx\s*=\s*(\d+).*?Nz\s*=\s*(\d+)", re.I)
_FSP_COLMAP = {"lat": "lat", "lon": "lon", "z": "depth", "slip": "slip", "rake": "rake",
               "trup": "t_rup", "rise": "t_rise", "sf_moment": "mo", "strike": "strike", "dip": "dip"}


def read_fsp(text: str) -> FiniteFaultModel:
    segs: List[Segment] = []
    rows: List[List[List[float]]] = []
    columns = None
    g_strike = g_dip = None
    g_dx = g_dz = None
    g_nx = g_nz = None
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            body = line.lstrip("%").strip()
            m = _FSP_SEG_RE.search(body)
            if m:
                cur = Segment(index=int(m.group(1)), seg_strike=float(m.group(2)), seg_dip=float(m.group(3)))
                segs.append(cur)
                rows.append([])
                continue
            m = _FSP_MECH_RE.search(body)
            if m and g_strike is None:
                g_strike, g_dip = float(m.group(1)), float(m.group(2))
            m = _FSP_DXDZ_RE.search(body)
            if m:
                if cur is not None:
                    cur.dx_km, cur.dy_km = float(m.group(1)), float(m.group(2))
                else:
                    g_dx, g_dz = float(m.group(1)), float(m.group(2))
            m = _FSP_NXNZ_RE.search(body)
            if m:
                if cur is not None:
                    cur.nx, cur.ny = int(m.group(1)), int(m.group(2))
                else:
                    g_nx, g_nz = int(m.group(1)), int(m.group(2))
            toks = body.split()
            if toks and toks[0].upper() == "LAT" and any(t.upper() == "SLIP" for t in toks):
                columns = [_FSP_COLMAP.get(t.lower().replace("==ew", "").replace("==ns", ""), t.lower()) for t in toks]
            continue
        nums = _numbers(line)
        if nums is None or columns is None:
            continue
        if cur is None:
            cur = Segment(index=1)
            segs.append(cur)
            rows.append([])
        if len(nums) == len(columns):
            rows[-1].append(nums)
    if columns is None or not any(rows):
        raise FFMFormatError("no subfault table found in FSP file")
    out = []
    for s, r in zip(segs, rows):
        if not r:
            continue
        d = np.array(r)
        col = {name: d[:, i] for i, name in enumerate(columns)}
        s.lat, s.lon, s.depth = col["lat"], col["lon"], col["depth"]
        s.slip_raw, s.rake = col["slip"], col["rake"]
        st = s.seg_strike if s.seg_strike is not None else g_strike
        dp = s.seg_dip if s.seg_dip is not None else g_dip
        if "strike" in col:
            s.strike, s.dip = col["strike"] % 360, col["dip"]
        else:
            if st is None or dp is None:
                raise FFMFormatError("FSP file gives no strike/dip")
            s.strike, s.dip = np.full(len(d), st % 360.0), np.full(len(d), dp)
        s.t_rup, s.t_rise = col.get("t_rup"), col.get("t_rise")
        s.moment_raw = col.get("mo")
        if s.dx_km is None:
            s.dx_km, s.dy_km = g_dx, g_dz
        if s.nx is None and g_nx and len(segs) == 1:
            s.nx, s.ny = g_nx, g_nz
        out.append(s)
    return FiniteFaultModel(segments=out, source_format="fsp")


def read_model(text: str, filename: str = "") -> FiniteFaultModel:
    name = filename.lower()
    if name.endswith(".fsp"):
        return read_fsp(text)
    if name.endswith(".param"):
        return read_param(text)
    # sniff
    if text.lstrip().startswith("%"):
        return read_fsp(text)
    return read_param(text)


# --------------------------------------------------------------------------------
# unit resolution
# --------------------------------------------------------------------------------
_SLIP_UNITS = {"cm": 0.01, "m": 1.0}
_MOM_UNITS = {"dyne-cm": 1e-7, "N-m": 1.0}


def resolve_units(model: FiniteFaultModel, ref_moment_nm: Optional[float] = None,
                  ref_max_slip_m: Optional[float] = None) -> FiniteFaultModel:
    """Determine slip and moment units and convert the model to SI in place.

    Decision order:
    1. reference maximum slip from the USGS product (most direct),
    2. physical plausibility of the implied rigidity mu = M0 / (A * D),
    3. reference scalar moment from the USGS product,
    4. format defaults (param: cm / dyne-cm, fsp: m / N-m).
    """
    raw_slip = np.concatenate([s.slip_raw for s in model.segments])
    max_raw = float(np.nanmax(raw_slip))
    have_mo = all(s.moment_raw is not None for s in model.segments)
    have_area = all(s.dx_km and s.dy_km for s in model.segments)
    default = ("cm", "dyne-cm") if model.source_format == "param" else ("m", "N-m")

    # --- slip unit
    slip_unit = None
    if ref_max_slip_m and max_raw > 0:
        ratios = {u: max_raw * f / ref_max_slip_m for u, f in _SLIP_UNITS.items()}
        best = min(ratios, key=lambda u: abs(math.log(ratios[u])))
        if 0.8 < ratios[best] < 1.25:
            slip_unit = best
            model.notes.append(f"slip unit '{best}' fixed by USGS maximum-slip ({ref_max_slip_m:.2f} m)")

    mom_unit = None
    if have_mo and have_area:
        # rigidity test over all unit combinations
        candidates = []
        for su, sf in _SLIP_UNITS.items():
            if slip_unit and su != slip_unit:
                continue
            for mu_, mf in _MOM_UNITS.items():
                vals = []
                for s in model.segments:
                    ok = s.slip_raw > 0.05 * max_raw
                    if not np.any(ok):
                        continue
                    a = model.subfault_area_m2(s)
                    vals.append((s.moment_raw[ok] * mf) / (a * s.slip_raw[ok] * sf))
                if not vals:
                    continue
                rig = float(np.median(np.concatenate(vals)))
                candidates.append((su, mu_, rig))
        plausible = [c for c in candidates if 5e9 <= c[2] <= 1.5e11]
        if len(plausible) == 1:
            slip_unit, mom_unit, model.rigidity_pa = plausible[0]
            model.notes.append(f"units from implied rigidity ({model.rigidity_pa / 1e9:.0f} GPa)")
        elif plausible and ref_moment_nm:
            def score(c):
                tot = sum(np.nansum(s.moment_raw) for s in model.segments) * _MOM_UNITS[c[1]]
                return abs(math.log(tot / ref_moment_nm))
            slip_unit, mom_unit, model.rigidity_pa = min(plausible, key=score)
            model.notes.append("units from rigidity + USGS scalar moment")
        elif plausible:
            pick = [c for c in plausible if (c[0], c[1]) == default] or plausible
            slip_unit, mom_unit, model.rigidity_pa = pick[0]
            model.notes.append("units ambiguous; format default used")
    if mom_unit is None and have_mo and ref_moment_nm:
        tot = sum(np.nansum(s.moment_raw) for s in model.segments)
        mom_unit = min(_MOM_UNITS, key=lambda u: abs(math.log(tot * _MOM_UNITS[u] / ref_moment_nm)))
    if slip_unit is None:
        slip_unit = default[0]
        model.notes.append(f"slip unit assumed '{slip_unit}' (format default)")
    if mom_unit is None and have_mo:
        mom_unit = default[1]
        model.notes.append(f"moment unit assumed '{mom_unit}' (format default)")

    model.slip_unit, model.moment_unit = slip_unit, (mom_unit or "n/a")
    for s in model.segments:
        s.slip = s.slip_raw * _SLIP_UNITS[slip_unit]
        if s.moment_raw is not None:
            s.moment = s.moment_raw * _MOM_UNITS[mom_unit]
        elif model.rigidity_pa or have_area:
            mu = model.rigidity_pa or 3.0e10
            s.moment = mu * model.subfault_area_m2(s) * s.slip
            model.notes.append(f"segment {s.index}: moment computed with mu={mu / 1e9:.0f} GPa")
    if model.rigidity_pa is None and have_area and model.total_moment:
        area_slip = sum(model.subfault_area_m2(s) * np.nansum(s.slip) for s in model.segments)
        if area_slip > 0:
            model.rigidity_pa = model.total_moment / area_slip
    return model


# --------------------------------------------------------------------------------
# moment-rate function
# --------------------------------------------------------------------------------
def read_moment_rate(text: str, ref_moment_nm: Optional[float] = None):
    """Return (t [s], rate [N-m/s], unit_note). Unit fixed by integrating."""
    t, r = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#%":
            continue
        nums = _numbers(line)
        if nums and len(nums) >= 2:
            t.append(nums[0])
            r.append(nums[1])
    if len(t) < 3:
        raise FFMFormatError("moment-rate file contains fewer than 3 samples")
    t, r = np.asarray(t), np.asarray(r)
    order = np.argsort(t)
    t, r = t[order], r[order]
    note = "assumed N-m/s"
    if ref_moment_nm:
        integral = float(_trapezoid(r, t))
        if integral > 0:
            f = min((1.0, 1e-7), key=lambda f: abs(math.log(integral * f / ref_moment_nm)))
            r = r * f
            note = "dyne-cm/s converted" if f != 1.0 else "N-m/s"
    return t, r, note


def mw_from_m0(m0_nm: float) -> float:
    """Moment magnitude (Hanks & Kanamori 1979; IASPEI 2013 constant 9.1)."""
    return (2.0 / 3.0) * (math.log10(m0_nm) - 9.1)
