"""Access to USGS ComCat (event detail feed, product files, FDSN event search).

Every downloaded file is cached in ``<cache>/<eventid>/`` so that a figure
can be re-made offline (``--offline``) and so that the exact product
versions used for a figure are archived next to it.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

DETAIL_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/{eid}.geojson"
FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USER_AGENT = "ffviz/1.0 (finite-fault figure tool; python-requests)"
TIMEOUT = 60


class DataError(RuntimeError):
    pass


# --------------------------------------------------------------------------------
# low-level I/O
# --------------------------------------------------------------------------------
def _http_get(url: str, params=None) -> bytes:
    import requests
    r = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    if r.status_code == 204:
        return b""
    if r.status_code != 200:
        raise DataError(f"HTTP {r.status_code} for {r.url}")
    return r.content


def _cached(path: str, fetch, offline: bool, refresh: bool) -> bytes:
    """Return cached data, or download it.

    refresh=True means "try to download a fresh copy"; if that fails (no
    internet, USGS down) the cached copy is used instead and a warning printed.
    """
    have = os.path.exists(path)
    if have and (not refresh or offline):
        with open(path, "rb") as f:
            return f.read()
    if offline:
        raise DataError(f"offline mode: {path} not in cache")
    try:
        data = fetch()
    except Exception as exc:
        if have:
            print(f"warning: could not update {os.path.basename(path)} ({exc}); using cached copy")
            with open(path, "rb") as f:
                return f.read()
        raise DataError(str(exc)) from exc
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def _decode(data: bytes) -> str:
    if data[:2] == b"\x1f\x8b":           # gzip-compressed file
        import gzip
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------------
# event + products
# --------------------------------------------------------------------------------
@dataclass
class Product:
    type: str
    code: str
    source: str
    update_time: int
    weight: int
    properties: Dict[str, str]
    contents: Dict[str, dict]

    def prop(self, key, cast=float, default=None):
        v = self.properties.get(key)
        if v in (None, ""):
            return default
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default

    @property
    def updated(self) -> str:
        return datetime.fromtimestamp(self.update_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def find_file(self, *suffixes) -> Optional[str]:
        for suf in suffixes:
            for name in self.contents:
                if name.lower().endswith(suf):
                    return name
        return None


@dataclass
class Event:
    id: str
    lon: float
    lat: float
    depth: float
    time: datetime
    mag: float
    mag_type: str
    place: str
    title: str
    products: Dict[str, List[Product]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def preferred(self, ptype: str) -> Optional[Product]:
        lst = self.products.get(ptype) or []
        if not lst:
            return None
        return sorted(lst, key=lambda p: (p.weight, p.update_time), reverse=True)[0]


def parse_event(detail: dict) -> Event:
    p = detail["properties"]
    lon, lat, dep = detail["geometry"]["coordinates"][:3]
    ev = Event(
        id=detail.get("id") or p.get("code"),
        lon=float(lon), lat=float(lat), depth=float(dep if dep is not None else float("nan")),
        time=datetime.fromtimestamp(p["time"] / 1000.0, tz=timezone.utc),
        mag=float(p.get("mag") or float("nan")), mag_type=p.get("magType") or "",
        place=p.get("place") or "", title=p.get("title") or "",
    )
    for ptype, lst in (p.get("products") or {}).items():
        ev.products[ptype] = [
            Product(type=ptype, code=x.get("code", ""), source=x.get("source", ""),
                    update_time=int(x.get("updateTime") or 0),
                    weight=int(x.get("preferredWeight") or 0),
                    properties=x.get("properties") or {}, contents=x.get("contents") or {})
            for x in lst if x.get("status", "UPDATE").upper() != "DELETE"
        ]
    return ev


def _ffm_version(ev: Optional["Event"]):
    p = ev.preferred("finite-fault") if ev else None
    return (p.code, p.update_time) if p else None


def load_event(eid: str, cache: str, offline=False, refresh=False) -> Event:
    """Event detail. Online, it is always re-downloaded (small file) so that
    new USGS product versions are picked up automatically; the large product
    files are cached per version and only downloaded when the version changes."""
    path = os.path.join(cache, eid, "event.geojson")
    old = None
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                old = parse_event(json.loads(_decode(f.read())))
        except Exception:
            old = None
    raw = _cached(path, lambda: _http_get(DETAIL_URL.format(eid=eid)), offline, refresh=not offline)
    ev = parse_event(json.loads(_decode(raw)))
    v_old, v_new = _ffm_version(old), _ffm_version(ev)
    if v_old and v_new and v_old != v_new:
        ev.notes.append(f"USGS updated the finite-fault model since the last run "
                        f"({v_old[0]} -> {v_new[0]}, updated {ev.preferred('finite-fault').updated})")
    return ev


def load_product_file(ev: Event, prod: Product, name: str, cache: str, offline=False, refresh=False) -> str:
    tag = f"{prod.type}_{prod.source}_{prod.code}_{prod.update_time}"
    path = os.path.join(cache, ev.id, tag, name.replace("/", "_"))
    url = prod.contents[name].get("url")
    if not url and not os.path.exists(path):
        raise DataError(f"no URL for {name}")
    return _decode(_cached(path, lambda: _http_get(url), offline, refresh))


# --------------------------------------------------------------------------------
# moment tensor
# --------------------------------------------------------------------------------
@dataclass
class MomentTensor:
    source: str
    method: str
    mw: Optional[float]
    m0: Optional[float]
    np1: Optional[tuple]
    np2: Optional[tuple]
    tensor: Optional[list]          # [mrr, mtt, mpp, mrt, mrp, mtp] in N-m
    depth: Optional[float]
    pct_dc: Optional[float]
    lat: Optional[float]
    lon: Optional[float]
    updated: str


def moment_tensor(ev: Event) -> Optional[MomentTensor]:
    p = ev.preferred("moment-tensor")
    if p is None:
        return None
    g = p.prop

    def plane(k):
        vals = [g(f"nodal-plane-{k}-{x}") for x in ("strike", "dip", "rake")]
        return tuple(vals) if None not in vals else None
    comps = [g(f"tensor-{c}") for c in ("mrr", "mtt", "mpp", "mrt", "mrp", "mtp")]
    pct = g("percent-double-couple")
    if pct is not None and pct <= 1.0:
        pct *= 100.0
    return MomentTensor(
        source=p.source, method=p.properties.get("derived-magnitude-type") or p.properties.get("beachball-type", ""),
        mw=g("derived-magnitude"), m0=g("scalar-moment"), np1=plane(1), np2=plane(2),
        tensor=None if None in comps else comps, depth=g("derived-depth"), pct_dc=pct,
        lat=g("derived-latitude"), lon=g("derived-longitude"), updated=p.updated)


# --------------------------------------------------------------------------------
# aftershock catalogue (FDSN, CSV)
# --------------------------------------------------------------------------------
FIXED_DEPTHS = (0.0, 5.0, 10.0, 33.0, 35.0)


@dataclass
class Catalog:
    time: np.ndarray        # days relative to mainshock
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray
    mag: np.ndarray
    mag_type: List[str]
    ids: List[str]
    fixed_depth: np.ndarray  # bool: depth equals a typical default value
    query: dict

    def __len__(self):
        return len(self.lat)


def load_catalog(ev: Event, cache: str, radius_km: float, days_before: float, days_after: float,
                 minmag: float, offline=False, refresh=False) -> Catalog:
    t0 = ev.time - timedelta(days=days_before)
    t_end = ev.time + timedelta(days=days_after)
    now = datetime.now(timezone.utc)
    complete = t_end <= now                  # window closed -> cached file is final
    t1 = min(t_end, now)
    q = dict(format="csv", starttime=t0.strftime("%Y-%m-%dT%H:%M:%S"),
             endtime=t1.strftime("%Y-%m-%dT%H:%M:%S"), latitude=f"{ev.lat:.4f}",
             longitude=f"{ev.lon:.4f}", maxradiuskm=f"{radius_km:.0f}",
             minmagnitude=f"{minmag:.1f}", orderby="time-asc", limit="20000", eventtype="earthquake")
    key = f"catalog_r{q['maxradiuskm']}_m{q['minmagnitude']}_b{days_before:g}_a{days_after:g}.csv"
    path = os.path.join(cache, ev.id, key)
    # an open window (end in the future) is re-downloaded unless we are offline
    raw = _cached(path, lambda: _http_get(FDSN_URL, q), offline, refresh or (not complete and not offline))
    fetched = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    q["endtime"] = min(t_end, fetched).strftime("%Y-%m-%dT%H:%M:%S")   # what the file really covers
    rows =list(csv.DictReader(io.StringIO(_decode(raw))))
    keep = [r for r in rows if r.get("id") != ev.id and r.get("latitude")]
    def f(r, k):
        try:
            return float(r[k])
        except (KeyError, TypeError, ValueError):
            return float("nan")
    times = []
    for r in keep:
        tt = datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
        times.append((tt - ev.time).total_seconds() / 86400.0)
    depth = np.array([f(r, "depth") for r in keep])
    fixed = np.array([any(abs(d - fd) < 1e-3 for fd in FIXED_DEPTHS) for d in depth], dtype=bool)
    return Catalog(time=np.array(times), lat=np.array([f(r, "latitude") for r in keep]),
                   lon=np.array([f(r, "longitude") for r in keep]), depth=depth,
                   mag=np.array([f(r, "mag") for r in keep]), mag_type=[r.get("magType", "") for r in keep],
                   ids=[r.get("id", "") for r in keep], fixed_depth=fixed, query=q)
