"""Synthetic events written in the USGS formats, for offline tests.

The geometry here is computed with a *spherical* flat-earth approximation
that is independent of ffviz.geometry (which uses pyproj on WGS84), so the
outline check in ffviz really tests the geometry code.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

R_EARTH = 6371.0


def _offset(lat, lon, de, dn):
    """Move by de (east) / dn (north) km on a sphere (small distances)."""
    lat2 = lat + math.degrees(dn / R_EARTH)
    lon2 = lon + math.degrees(de / (R_EARTH * math.cos(math.radians(0.5 * (lat + lat2)))))
    return lat2, lon2


def _sdr_to_mt(strike, dip, rake, m0):
    """Moment tensor (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp), Aki & Richards / GCMT."""
    f, d, l = map(math.radians, (strike, dip, rake))
    mrr = m0 * math.sin(2 * d) * math.sin(l)
    mtt = -m0 * (math.sin(d) * math.cos(l) * math.sin(2 * f) + math.sin(2 * d) * math.sin(l) * math.sin(f) ** 2)
    mpp = m0 * (math.sin(d) * math.cos(l) * math.sin(2 * f) - math.sin(2 * d) * math.sin(l) * math.cos(f) ** 2)
    mrt = -m0 * (math.cos(d) * math.cos(l) * math.cos(f) + math.cos(2 * d) * math.sin(l) * math.sin(f))
    mrp = m0 * (math.cos(d) * math.cos(l) * math.sin(f) - math.cos(2 * d) * math.sin(l) * math.cos(f))
    mtp = -m0 * (math.sin(d) * math.cos(l) * math.cos(2 * f) + 0.5 * math.sin(2 * d) * math.sin(l) * math.sin(2 * f))
    return mrr, mtt, mpp, mrt, mrp, mtp


def _segment(hlat, hlon, hdep, strike, dip, nx, ny, dx, dy, hi, hj, slipfun, rake, vr=2.0, t0=0.0, href=None):
    """Return header text, rows and stats for one segment.

    (hi, hj) is the 1-based cell containing the reference point (hlat, hlon, hdep).
    href: (lat, lon, dep) of the true hypocentre for rupture times.
    """
    phi, dl = math.radians(strike), math.radians(dip)
    s = np.array([math.sin(phi), math.cos(phi), 0.0])
    d = np.array([math.cos(phi) * math.cos(dl), -math.sin(phi) * math.cos(dl), math.sin(dl)])
    rows, cells = [], []
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            off = (i - hi) * dx * s + (j - hj) * dy * d
            la, lo = _offset(hlat, hlon, off[0], off[1])
            z = hdep + off[2]
            cells.append((i, j, la, lo, z, off))
    # corners of the segment outline
    def corner(ci, cj):
        off = (ci - hi) * dx * s + (cj - hj) * dy * d
        la, lo = _offset(hlat, hlon, off[0], off[1])
        return lo, la, hdep + off[2]
    bnd = [corner(0.5, 0.5), corner(nx + 0.5, 0.5), corner(nx + 0.5, ny + 0.5), corner(0.5, ny + 0.5), corner(0.5, 0.5)]
    href = href or (hlat, hlon, hdep)
    mu = 3.0e11  # dyne/cm^2
    area_cm2 = dx * dy * 1e10
    for i, j, la, lo, z, off in cells:
        a, w = (i - hi) * dx, (j - hj) * dy
        slip_m = slipfun(a, w)
        de = (lo - href[1]) * math.radians(1) * R_EARTH * math.cos(math.radians(la))
        dn = (la - href[0]) * math.radians(1) * R_EARTH
        dist = math.sqrt(de ** 2 + dn ** 2 + (z - href[2]) ** 2)
        rk = rake(a, w) if callable(rake) else rake
        mo = mu * area_cm2 * slip_m * 100.0          # dyne-cm
        rows.append((la, lo, z, slip_m * 100.0, rk, strike, dip, t0 + dist / vr, 2.0, 2.0, mo))
    return bnd, rows


def _param_text(segments):
    out = [f"#Total number of fault_segments=     {len(segments)}"]
    for k, (nx, ny, dx, dy, hi, hj, hlon, hlat, bnd, rows) in enumerate(segments, start=1):
        out.append(f"#Fault_segment =   {k} nx(Along-strike)=  {nx:3d} Dx = {dx:5.2f}km ny(downdip)=  {ny:3d} Dy = {dy:5.2f}km")
        out.append(f"#Boundary of Fault_segment     {k}. EQ in cell ({hi},{hj}). Lon:  {hlon:9.4f}   Lat:  {hlat:8.4f}")
        out.append("#Lon.  Lat.  Depth")
        for lo, la, z in bnd:
            out.append(f"  {lo:10.5f} {la:9.5f} {z:9.5f}")
        out.append("#Lat. Lon. depth slip rake strike dip t_rup t_ris t_fal mo")
        for r in rows:
            out.append("  {:9.5f} {:10.5f} {:9.5f} {:9.4f} {:8.4f} {:8.3f} {:7.3f} {:8.4f} {:8.4f} {:8.4f} {:.4e}".format(*r))
    return "\n".join(out) + "\n"


def _fsp_text(segments, strikes, dips):
    out = ["% ---------------------------------- FINITE-SOURCE RUPTURE MODEL --------------------------------",
           "% Event : synthetic", f"% Mech : STRK = {strikes[0]:.1f} DIP = {dips[0]:.1f} RAKE = 0"]
    for k, ((nx, ny, dx, dy, hi, hj, hlon, hlat, bnd, rows), st, dp) in enumerate(zip(segments, strikes, dips), 1):
        out.append(f"% SEGMENT # {k}: STRIKE = {st:.1f} deg DIP = {dp:.1f} deg")
        out.append(f"% Dx = {dx:.2f} km Dz = {dy:.2f} km")
        out.append(f"% Nx = {nx} Nz = {ny}")
        out.append("% LAT LON X==EW Y==NS Z SLIP RAKE TRUP RISE SF_MOMENT")
        for r in rows:
            out.append(f"{r[0]:9.4f} {r[1]:10.4f} 0.0 0.0 {r[2]:8.3f} {r[3] / 100:8.4f} {r[4]:7.2f} {r[7]:7.2f} {r[8]:5.2f} {r[10] * 1e-7:.4e}")
    return "\n".join(out) + "\n"


def _stf_text(total_dyne_cm, dur=33.0):
    t = np.arange(0, 45.01, 0.5)
    shape = np.where(t < dur, np.sin(np.pi * t / dur) ** 2, 0.0)
    shape *= total_dyne_cm / np.trapezoid(shape, t) if hasattr(np, "trapezoid") else total_dyne_cm / np.trapz(shape, t)
    return "\n".join(f"{a:8.2f} {b:.5e}" for a, b in zip(t, shape)) + "\n"


def _catalog_csv(ev_time, lat, lon, strike, n=60, seed=1, eid="synthetic"):
    rng = np.random.default_rng(seed)
    hdr = "time,latitude,longitude,depth,mag,magType,nst,gap,dmin,rms,net,id,updated,place,type,horizontalError,depthError,magError,magNst,status,locationSource,magSource"
    lines = [hdr]
    lines.append(f"{ev_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')},{lat},{lon},12.6,7.7,mww,,,,,us,{eid},,mainshock,earthquake,,,,,reviewed,us,us")
    phi = math.radians(strike)
    for k in range(n):
        a = rng.uniform(-80, 70)
        w = rng.normal(5, 8)
        de, dn = a * math.sin(phi) + w * math.cos(phi), a * math.cos(phi) - w * math.sin(phi)
        la, lo = _offset(lat, lon, de, dn)
        lo = (lo + 180) % 360 - 180
        dep = 10.0 if rng.random() < 0.4 else float(rng.uniform(4, 30))
        t = ev_time + timedelta(hours=float(rng.exponential(48)))
        m = 4.5 + rng.exponential(0.5)
        lines.append(f"{t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z,{la:.4f},{lo:.4f},{dep:.3f},{m:.1f},mb,,,,,us,us_as{k},,place,earthquake,,,,,reviewed,us,us")
    return "\n".join(lines) + "\n"


def write_event(cache, kind="panama"):
    eid = {"panama": "test_panama", "two_segment": "test_twoseg", "dateline": "test_dateline"}[kind]
    ev_time = datetime(2026, 1, 9, 17, 56, 6, tzinfo=timezone.utc)
    if kind == "dateline":
        hlat, hlon, hdep = -15.5, 179.95, 15.0
    else:
        hlat, hlon, hdep = 7.587, -80.769, 12.6
    if kind in ("panama", "dateline"):
        def slip(a, w):
            return 7.8 * math.exp(-((a - 25) / 12) ** 2 / 2 - ((w - (-3)) / 6) ** 2 / 2) + \
                5.5 * math.exp(-((a + 28) / 12) ** 2 / 2 - ((w - 2) / 8) ** 2 / 2)
        rake = lambda a, w: -30.0 if a < 0 else 0.0
        nx, ny, dx, dy, hi, hj = 25, 8, 6.0, 4.5, 15, 4
        bnd, rows = _segment(hlat, hlon, hdep, 288.0, 50.0, nx, ny, dx, dy, hi, hj, slip, rake)
        segs = [(nx, ny, dx, dy, hi, hj, hlon, hlat, bnd, rows)]
        strikes, dips = [288.0], [50.0]
        mt_sdr = (288.0, 66.0, -17.0)
    else:
        def slip1(a, w):
            return 8.5 * math.exp(-((a - 5) / 15) ** 2 / 2 - ((w - 5) / 12) ** 2 / 2)
        def slip2(a, w):
            return 3.0 * math.exp(-((a - 0) / 10) ** 2 / 2 - ((w - 0) / 6) ** 2 / 2)
        b1, r1 = _segment(hlat, hlon, 55.0, 357.0, 51.0, 17, 13, 6.0, 6.0, 9, 7, slip1, 103.0)
        la2, lo2 = _offset(hlat, hlon, -10.0, 60.0)
        b2, r2 = _segment(la2, lo2, 20.0, 20.0, 30.0, 8, 6, 5.0, 5.0, 4, 3, slip2, 95.0, t0=20.0,
                          href=(hlat, hlon, 55.0))
        segs = [(17, 13, 6.0, 6.0, 9, 7, hlon, hlat, b1, r1), (8, 6, 5.0, 5.0, 4, 3, lo2, la2, b2, r2)]
        strikes, dips = [357.0, 20.0], [51.0, 30.0]
        hdep = 55.0
        mt_sdr = (160.6, 44.6, 76.8)

    total_dyne = sum(r[10] for s in segs for r in s[9])
    m0 = total_dyne * 1e-7
    mw = 2 / 3 * (math.log10(m0) - 9.1)
    max_slip = max(r[3] for s in segs for r in s[9]) / 100.0

    from ffviz.beachball import aux_plane, sdr_to_tensor
    s2, d2, r2_ = aux_plane(*mt_sdr)
    tensor = sdr_to_tensor(*mt_sdr, m0)
    upd = 1791569140040
    d = os.path.join(cache, eid)
    ffdir = os.path.join(d, f"finite-fault_us_{eid}_{upd}")
    os.makedirs(ffdir, exist_ok=True)
    with open(os.path.join(ffdir, "basic_inversion.param"), "w") as f:
        f.write(_param_text(segs))
    with open(os.path.join(ffdir, "complete_inversion.fsp"), "w") as f:
        f.write(_fsp_text(segs, strikes, dips))
    with open(os.path.join(ffdir, "moment_rate.mr"), "w") as f:
        f.write(_stf_text(total_dyne))
    det = {
        "type": "Feature", "id": eid,
        "geometry": {"type": "Point", "coordinates": [hlon, hlat, hdep]},
        "properties": {
            "mag": round(mw, 1), "magType": "mww", "time": int(ev_time.timestamp() * 1000),
            "place": f"Synthetic test event ({kind})", "title": f"M {mw:.1f} - synthetic {kind}", "code": eid,
            "products": {
                "finite-fault": [{
                    "code": eid, "source": "us", "updateTime": upd, "preferredWeight": 6, "status": "UPDATE",
                    "properties": {"scalar-moment": f"{m0 * 1.01:.4e}", "maximum-slip": f"{max_slip:.4f}",
                                   "latitude": str(hlat), "longitude": str(hlon), "depth": str(hdep),
                                   "average-rupture-velocity": "2.0", "segments": str(len(segs))},
                    "contents": {n: {"url": ""} for n in ("basic_inversion.param", "complete_inversion.fsp", "moment_rate.mr")},
                }],
                "moment-tensor": [{
                    "code": eid + "_mww", "source": "us", "updateTime": upd, "preferredWeight": 16, "status": "UPDATE",
                    "properties": {
                        "derived-magnitude": f"{mw:.2f}", "derived-magnitude-type": "Mww", "scalar-moment": f"{m0:.4e}",
                        "nodal-plane-1-strike": str(mt_sdr[0]), "nodal-plane-1-dip": str(mt_sdr[1]),
                        "nodal-plane-1-rake": str(mt_sdr[2]), "nodal-plane-2-strike": f"{s2:.2f}",
                        "nodal-plane-2-dip": f"{d2:.2f}", "nodal-plane-2-rake": f"{r2_:.2f}",
                        "derived-depth": "13.5", "percent-double-couple": "0.78",
                        **{f"tensor-{c}": f"{v:.4e}" for c, v in zip(("mrr", "mtt", "mpp", "mrt", "mrp", "mtp"), tensor)},
                    },
                    "contents": {},
                }],
            },
        },
    }
    with open(os.path.join(d, "event.geojson"), "w") as f:
        json.dump(det, f)
    return eid, dict(m0=m0, max_slip=max_slip, mw=mw, segs=segs, hyp=(hlat, hlon, hdep), ev_time=ev_time,
                     strikes=strikes)


def write_catalog(cache, eid, info, radius, minmag, days_before=0.0, days_after=30.0):
    path = os.path.join(cache, eid, f"catalog_r{radius:.0f}_m{minmag:.1f}_b{days_before:g}_a{days_after:g}.csv")
    lat, lon, _ = info["hyp"]
    with open(path, "w") as f:
        f.write(_catalog_csv(info["ev_time"], lat, lon, info["strikes"][0], eid=eid))
    return path
