"""Fault geometry: subfault corners, fault-plane coordinates, profiles.

All horizontal computations use a local azimuthal-equidistant projection
centred on the hypocentre (pyproj, WGS84), so distances are accurate to well
below 0.1 % within a few hundred km of the source.

Conventions (Aki & Richards): strike measured clockwise from north, the
fault dips to the right of the strike direction, rake measured in the fault
plane anticlockwise from the strike direction (hanging-wall motion).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pyproj import Transformer

from .ffm import FiniteFaultModel, Segment


def wrap_lon(lon, lon0):
    """Wrap longitudes into [lon0-180, lon0+180) (dateline-safe)."""
    return (np.asarray(lon, dtype=float) - lon0 + 180.0) % 360.0 - 180.0 + lon0


class LocalFrame:
    """Local Cartesian frame (km, x=east, y=north) around (lat0, lon0)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        proj = (f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 "
                f"+ellps=WGS84 +units=m +no_defs")
        self._fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        self._inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0

    def to_lonlat(self, x_km, y_km):
        lon, lat = self._inv.transform(np.asarray(x_km, float) * 1000.0, np.asarray(y_km, float) * 1000.0)
        return wrap_lon(lon, self.lon0), np.asarray(lat)


def strike_dip_vectors(strike_deg, dip_deg):
    """Unit vectors (east, north, down) along strike and down dip."""
    phi, dl = np.radians(strike_deg), np.radians(dip_deg)
    s = np.stack([np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
    d = np.stack([np.cos(phi) * np.cos(dl), -np.sin(phi) * np.cos(dl), np.sin(dl)], axis=-1)
    return s, d


def subfault_corners(seg: Segment, frame: LocalFrame):
    """Return lon (n,4), lat (n,4), depth (n,4) of each subfault rectangle.

    Corner order: top-start, top-end, bottom-end, bottom-start (top = up-dip,
    start/end = along strike).
    """
    if seg.dx_km is None or seg.dy_km is None:
        raise ValueError(f"segment {seg.index}: subfault size unknown")
    x, y = frame.to_xy(seg.lon, seg.lat)
    c = np.stack([x, y, seg.depth], axis=-1)
    s, d = strike_dip_vectors(seg.strike, seg.dip)
    hs, hd = 0.5 * seg.dx_km, 0.5 * seg.dy_km
    corners = np.stack([c - hs * s - hd * d, c + hs * s - hd * d,
                        c + hs * s + hd * d, c - hs * s + hd * d], axis=1)  # (n,4,3)
    lon, lat = frame.to_lonlat(corners[..., 0], corners[..., 1])
    return lon, lat, corners[..., 2]


@dataclass
class PlaneCoords:
    """Fault-plane coordinates of one segment (km)."""
    along: np.ndarray        # along-strike position of subfault centres
    downdip: np.ndarray      # down-dip distance from the segment top edge
    strike: float
    dip: float
    top_depth: float         # depth of the segment's upper edge (km)
    bottom_depth: float
    length: float
    width: float
    origin_xy: Tuple[float, float]   # local xy of the along=0 reference point (on top edge)


def plane_coords(seg: Segment, frame: LocalFrame, ref_xy=(0.0, 0.0)) -> PlaneCoords:
    """Along-strike / down-dip coordinates of subfault centres.

    along = projection of the horizontal offset from ``ref_xy`` on the strike
    direction; downdip = (depth - top) / sin(dip).  Using depth for the
    down-dip coordinate makes the result exact for planar segments.
    """
    phi, dip = seg.mean_strike, seg.mean_dip
    x, y = frame.to_xy(seg.lon, seg.lat)
    s_e, s_n = math.sin(math.radians(phi)), math.cos(math.radians(phi))
    along = (x - ref_xy[0]) * s_e + (y - ref_xy[1]) * s_n
    sd = math.sin(math.radians(dip))
    top = float(np.min(seg.depth) - 0.5 * seg.dy_km * sd)
    bot = float(np.max(seg.depth) + 0.5 * seg.dy_km * sd)
    downdip = (seg.depth - top) / sd
    length = float(np.ptp(along) + seg.dx_km)
    width = float(np.ptp(downdip) + seg.dy_km)
    return PlaneCoords(along, downdip, phi, dip, top, bot, length, width, ref_xy)


def segment_outline(seg: Segment, frame: LocalFrame):
    """Outline (lon, lat, depth) of the whole segment from the subfault grid."""
    lon, lat, dep = subfault_corners(seg, frame)
    pc = plane_coords(seg, frame)
    a0, a1 = pc.along.min(), pc.along.max()
    w0, w1 = pc.downdip.min(), pc.downdip.max()
    pts = []
    # pick the subfault corner closest to each of the four extreme grid corners
    targets = [(a0, w0, 0), (a1, w0, 1), (a1, w1, 2), (a0, w1, 3)]
    for a, w, k in targets:
        i = int(np.argmin((pc.along - a) ** 2 + (pc.downdip - w) ** 2))
        pts.append((float(lon[i, k]), float(lat[i, k]), float(dep[i, k])))
    return pts


def top_edge(seg: Segment, frame: LocalFrame):
    """(lon, lat) of the upper edge of the segment (two end points) and its depth."""
    out = segment_outline(seg, frame)
    return (out[0][0], out[0][1]), (out[1][0], out[1][1]), 0.5 * (out[0][2] + out[1][2])


def distance_to_plane_km(seg: Segment, frame: LocalFrame, lon, lat, depth):
    """Perpendicular distance (km) from a point to the best-fit segment plane."""
    x, y = frame.to_xy(seg.lon, seg.lat)
    c = np.stack([x, y, seg.depth], axis=-1).mean(axis=0)
    s, d = strike_dip_vectors(seg.mean_strike, seg.mean_dip)
    n = np.cross(s, d)
    px, py = frame.to_xy(lon, lat)
    p = np.array([float(px), float(py), float(depth)])
    return float(abs(np.dot(p - c, n)))


def profile_coords(frame: LocalFrame, lon, lat, strike_deg):
    """Distance along the dip-direction profile (p) and along strike (q), km."""
    x, y = frame.to_xy(lon, lat)
    phi = math.radians(strike_deg)
    p = x * math.cos(phi) - y * math.sin(phi)   # towards dip direction (strike + 90)
    q = x * math.sin(phi) + y * math.cos(phi)
    return p, q


def geodesic_point(lon, lat, azimuth_deg, dist_km):
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    lo, la, _ = g.fwd(lon, lat, azimuth_deg, dist_km * 1000.0)
    return lo, la


def extent_of(lons, lats, lon0, pad_frac=0.12, min_span_deg=1.0):
    lons = wrap_lon(lons, lon0)
    lo0, lo1 = float(np.nanmin(lons)), float(np.nanmax(lons))
    la0, la1 = float(np.nanmin(lats)), float(np.nanmax(lats))
    dlo, dla = max(lo1 - lo0, min_span_deg), max(la1 - la0, min_span_deg)
    cx, cy = 0.5 * (lo0 + lo1), 0.5 * (la0 + la1)
    dlo *= 1 + 2 * pad_frac
    dla *= 1 + 2 * pad_frac
    return [cx - dlo / 2, cx + dlo / 2, max(cy - dla / 2, -85.0), min(cy + dla / 2, 85.0)]
