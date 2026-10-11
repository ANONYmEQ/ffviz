"""Multi-panel figure: map, fault-plane view(s), cross-section, moment rate."""
from __future__ import annotations

import json
import math
import os
import warnings
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from . import __version__
from .ffm import FiniteFaultModel, mw_from_m0, _trapezoid
from .geometry import (LocalFrame, subfault_corners, plane_coords, segment_outline, profile_coords,
                       geodesic_point, wrap_lon, extent_of)

SLIP_CMAP = matplotlib.colormaps["inferno_r"].resampled(256)
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]
PLATES_URL = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json"


# --------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------
def _nice(x):
    """Round x down to 1, 2, 2.5 or 5 x 10^n."""
    if x <= 0:
        return 1
    e = 10 ** math.floor(math.log10(x))
    for m in (5, 2.5, 2, 1):
        if m * e <= x:
            return m * e
    return e


def _ne_available(res: str, category="physical", names=("coastline", "land"), warn=True) -> bool:
    """Make sure the Natural Earth files exist locally (cartopy downloads them once)."""
    try:
        import cartopy.io.shapereader as shpreader
        for n in names:
            shpreader.natural_earth(resolution=res, category=category, name=n)
        return True
    except Exception as exc:  # network or download problems
        if warn:
            warnings.warn(f"Natural Earth {res} {names} not available ({exc})")
        return False


def _plate_lines(cache_dir: str):
    path = os.path.join(cache_dir, "PB2002_boundaries.json")
    try:
        if not os.path.exists(path):
            import requests
            r = requests.get(PLATES_URL, timeout=60)
            r.raise_for_status()
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "wb") as f:
                f.write(r.content)
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
    except Exception as exc:
        warnings.warn(f"plate boundaries not available ({exc})")
        return []
    lines = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        parts = coords if geom.get("type") == "MultiLineString" else [coords]
        for part in parts:
            if len(part) >= 2:
                lines.append(np.asarray(part, float))
    return lines


def _beachball(ax_bb, mt, color):
    from .beachball import draw_beachball
    if mt.tensor:
        draw_beachball(ax_bb, m6=mt.tensor, color=color)
    else:
        draw_beachball(ax_bb, sdr=mt.np1, color=color)


# --------------------------------------------------------------------------------
# main figure
# --------------------------------------------------------------------------------
def make_figure(ev, mt, ffp, model: FiniteFaultModel, catalog, stf, checks, args, out_path):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    lat0, lon0 = ev.lat, ev.lon
    frame = LocalFrame(lat0, lon0)
    pc_proj = ccrs.PlateCarree()

    # ---- geometry
    seg_geo = []
    all_lon, all_lat = [lon0], [lat0]
    for s in model.segments:
        lon, lat, dep = subfault_corners(s, frame)
        seg_geo.append((lon, lat, dep))
        all_lon.extend(lon.ravel())
        all_lat.extend(lat.ravel())
    vmax = float(np.nanmax(model.slip))
    norm = Normalize(0, vmax)

    main = max(model.segments, key=lambda s: float(np.nansum(s.moment)) if s.moment is not None else s.n)
    main_pc = plane_coords(main, frame)
    half_len = 0.5 * max(plane_coords(s, frame).length for s in model.segments)

    # FFM hypocentre (model) and ComCat hypocentre
    hyp = (lon0, lat0, ev.depth)
    if ffp is not None and ffp.prop("latitude") is not None and ffp.prop("longitude") is not None:
        fdep = ffp.prop("depth")
        hyp_ffm = (ffp.prop("longitude"), ffp.prop("latitude"), fdep if fdep is not None else ev.depth)
    else:
        hyp_ffm = hyp

    # cross-section A-A' through the (model) hypocentre, perpendicular to the main segment
    sec_az = (main_pc.strike + 90.0) % 360.0
    pmax = max(60.0, main_pc.width * math.cos(math.radians(main_pc.dip)) + 40.0)
    a_lon, a_lat = geodesic_point(hyp_ffm[0], hyp_ffm[1], sec_az + 180, pmax * 0.5)
    b_lon, b_lat = geodesic_point(hyp_ffm[0], hyp_ffm[1], sec_az, pmax)
    all_lon.extend([a_lon, b_lon])
    all_lat.extend([a_lat, b_lat])

    # aftershocks in the map window
    cat_ok = catalog is not None and len(catalog) > 0
    if cat_ok:
        all_lon.extend(catalog.lon)
        all_lat.extend(catalog.lat)
    ext = extent_of(np.array(all_lon), np.array(all_lat), lon0, pad_frac=0.08, min_span_deg=1.0)
    span = max(ext[1] - ext[0], ext[3] - ext[2])
    res = args.coast_res or ("10m" if span < 6 else "50m" if span < 30 else "110m")

    # ---- figure layout
    FW, FH = 17.5, 11.0
    fig = plt.figure(figsize=(FW, FH), dpi=args.dpi)
    proj = ccrs.Mercator(central_longitude=lon0)
    corners = np.array([[ext[0], ext[2]], [ext[1], ext[2]], [ext[1], ext[3]], [ext[0], ext[3]]])
    xy = proj.transform_points(pc_proj, corners[:, 0], corners[:, 1])
    x0m, x1m, y0m, y1m = xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()
    # size the map axes to the data aspect so that no space is wasted
    map_top, max_w, max_h = 0.885, 0.53, 0.74
    aspect = (y1m - y0m) / (x1m - x0m)
    w = max_w
    h = w * FW * aspect / FH
    if h > max_h:
        h = max_h
        w = h * FH / (FW * aspect)
    map_left = 0.035
    axm = fig.add_axes([map_left, map_top - h, w, h], projection=proj)
    axm.set_xlim(x0m, x1m)
    axm.set_ylim(y0m, y1m)
    map_bottom = map_top - h

    axm.set_facecolor("#d8e6ef")
    if not args.no_coast and _ne_available(res):
        axm.add_feature(cfeature.NaturalEarthFeature("physical", "land", res), facecolor="#f1eee4",
                        edgecolor="none", zorder=0.5)
        axm.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", res), facecolor="none",
                        edgecolor="#56626b", linewidth=0.6, zorder=1)
        if _ne_available(res, "cultural", ("admin_0_boundary_lines_land",), warn=False):
            axm.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", res),
                            facecolor="none", edgecolor="#8a8a8a", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)
    if not args.no_plates:
        for ln in _plate_lines(args.cache_dir):
            lo = wrap_lon(ln[:, 0], lon0)
            jumps = np.where(np.abs(np.diff(lo)) > 90)[0]
            for part_lo, part_la in zip(np.split(lo, jumps + 1), np.split(ln[:, 1], jumps + 1)):
                axm.plot(part_lo, part_la, transform=pc_proj, color="#7d5a8c", lw=1.0, alpha=0.7, zorder=1.2)

    gl = axm.gridlines(crs=pc_proj, draw_labels=True, linewidth=0.4, color="white", alpha=0.8, zorder=1.1)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}

    # ---- slip patches
    for s, (lon, lat, dep) in zip(model.segments, seg_geo):
        verts = [np.column_stack([lon[i], lat[i]]) for i in range(s.n)]
        order = np.argsort(-s.depth)            # deeper first, shallower drawn on top
        pcoll = PolyCollection([verts[i] for i in order], array=s.slip[order], cmap=SLIP_CMAP, norm=norm,
                               edgecolors=(0.3, 0.3, 0.3, 0.25), linewidths=0.2, alpha=0.9,
                               transform=pc_proj, zorder=3)
        axm.add_collection(pcoll)
        ol = segment_outline(s, frame)
        olo = [p[0] for p in ol] + [ol[0][0]]
        ola = [p[1] for p in ol] + [ol[0][1]]
        axm.plot(olo, ola, transform=pc_proj, color="#222", lw=0.8, zorder=3.2)
        top_d = 0.5 * (ol[0][2] + ol[1][2])
        axm.plot([ol[0][0], ol[1][0]], [ol[0][1], ol[1][1]], transform=pc_proj, color="#b3121a",
                 lw=2.6, zorder=3.3, solid_capstyle="round", path_effects=[pe.Stroke(linewidth=4.2, foreground="white"), pe.Normal()])
        lab_lo, lab_la = ol[1][0], ol[1][1]
        axm.text(lab_lo, lab_la, f"  top {top_d:.1f} km" + (f" (seg {s.index})" if len(model.segments) > 1 else ""),
                 transform=pc_proj, fontsize=7.8, color="#b3121a", ha="left", va="center", zorder=6,
                 path_effects=HALO)
    mappable_slip = matplotlib.cm.ScalarMappable(norm=norm, cmap=SLIP_CMAP)

    # ---- profile line A-A' on the map
    axm.plot([a_lon, b_lon], [a_lat, b_lat], transform=ccrs.Geodetic(), color="#1d3557", lw=1.0,
             ls=(0, (6, 3)), zorder=4)
    for lo_, la_, t_ in ((a_lon, a_lat, "A"), (b_lon, b_lat, "A′")):
        axm.text(lo_, la_, t_, transform=pc_proj, fontsize=11, fontweight="bold", color="#1d3557",
                 ha="center", va="center", zorder=6, path_effects=HALO)

    # ---- aftershocks
    sizes = lambda m: 6 + 9 * np.clip(m - args.minmag + 0.3, 0.1, None) ** 2.2
    if cat_ok:
        after = catalog.time >= 0
        cvals = catalog.depth if args.color_by == "depth" else catalog.time
        cmap_as = "plasma_r" if args.color_by == "depth" else "viridis"
        if args.color_by == "depth":
            vmin_as, vmax_as = 0, max(10.0, float(np.nanmax(catalog.depth)))
        else:   # colour range = what the catalogue actually covers (window may still be open)
            vmin_as, vmax_as = 0, max(1.0, min(float(args.days), math.ceil(float(np.nanmax(catalog.time)))))
        sc = axm.scatter(catalog.lon[after], catalog.lat[after], s=sizes(catalog.mag[after]), c=cvals[after],
                         cmap=cmap_as, vmin=vmin_as, vmax=vmax_as, edgecolors="#111", linewidths=0.4,
                         alpha=0.85, transform=pc_proj, zorder=5)
        if np.any(~after):
            axm.scatter(catalog.lon[~after], catalog.lat[~after], s=sizes(catalog.mag[~after]), c="#9a9a9a",
                        edgecolors="#111", linewidths=0.4, transform=pc_proj, zorder=4.9)
    else:
        sc = None

    # ---- epicentres
    axm.scatter([lon0], [lat0], marker="*", s=420, c="white", edgecolors="#111", linewidths=1.2,
                transform=pc_proj, zorder=7)
    d_hyp = math.hypot(*frame.to_xy(hyp_ffm[0], hyp_ffm[1]))
    if d_hyp > 2.0:
        axm.scatter([hyp_ffm[0]], [hyp_ffm[1]], marker="+", s=120, c="#111", linewidths=1.4,
                    transform=pc_proj, zorder=7)

    # ---- beachball (upper-left corner, leader line to epicentre)
    if mt is not None and (mt.tensor or mt.np1):
        bb = axm.inset_axes([0.015, 0.80, 0.16, 0.16])
        _beachball(bb, mt, "#b3121a")
        bb.text(0, -1.25, f"{mt.method or 'MT'} {mt.mw:.2f}" if mt.mw else "MT", ha="center", va="top",
                fontsize=8.5, path_effects=HALO)
        x_epi, y_epi = proj.transform_point(lon0, lat0, pc_proj)
        con = ConnectionPatch(xyA=(0.0, -1.0), coordsA=bb.transData, xyB=(x_epi, y_epi), coordsB=axm.transData,
                              color="#333", lw=0.7, zorder=6)
        fig.add_artist(con)

    # ---- scale bar
    w_km = frame.to_xy(ext[1], 0.5 * (ext[2] + ext[3]))[0] - frame.to_xy(ext[0], 0.5 * (ext[2] + ext[3]))[0]
    L = _nice(abs(w_km) / 5.0)
    sx = ext[0] + 0.05 * (ext[1] - ext[0])
    sy = ext[2] + 0.04 * (ext[3] - ext[2])
    ex, ey = geodesic_point(sx, sy, 90.0, L)
    mx, my = geodesic_point(sx, sy, 90.0, L / 2)
    axm.plot([sx, mx], [sy, my], transform=pc_proj, color="#111", lw=4, solid_capstyle="butt", zorder=8)
    axm.plot([mx, ex], [my, ey], transform=pc_proj, color="white", lw=4, solid_capstyle="butt", zorder=8,
             path_effects=[pe.Stroke(linewidth=5.2, foreground="#111"), pe.Normal()])
    for lo_, lab in ((sx, "0"), (mx, f"{L / 2:g}"), (ex, f"{L:g} km")):
        axm.text(lo_, sy + 0.015 * (ext[3] - ext[2]), lab, transform=pc_proj, ha="center", va="bottom",
                 fontsize=8, zorder=8, path_effects=HALO)

    # ---- globe inset
    gax = fig.add_axes([0.905, 0.905, 0.065, 0.065], projection=ccrs.Orthographic(lon0, lat0))
    gax.set_global()
    gax.set_facecolor("#d8e6ef")
    if not args.no_coast and _ne_available("110m", names=("land",), warn=False):
        gax.add_feature(cfeature.LAND, facecolor="#cfc8b8", edgecolor="none")
    gax.scatter([lon0], [lat0], transform=pc_proj, s=40, c="#b3121a", marker="*", zorder=5)
    gax.spines["geo"].set_linewidth(0.6)

    # ---- legend (map symbols)
    handles = [Line2D([], [], marker="*", ls="", ms=13, mfc="white", mec="#111", label="Epicentre (ComCat)"),
               Line2D([], [], color="#b3121a", lw=2.6, label="Upper edge of fault segment"),
               Line2D([], [], color="#1d3557", lw=1.0, ls=(0, (6, 3)), label="Cross-section A–A′")]
    if d_hyp > 2.0:
        handles.insert(1, Line2D([], [], marker="+", ls="", ms=10, mec="#111", mew=1.4, label="FFM hypocentre"))
    if not args.no_plates:
        handles.append(Line2D([], [], color="#7d5a8c", lw=1.0, label="Plate boundary (Bird 2003)"))
    if cat_ok:
        for mm in sorted({math.floor(args.minmag), math.floor(args.minmag) + 1, math.floor(args.minmag) + 2}):
            if mm >= args.minmag - 0.01 or mm == math.floor(args.minmag):
                handles.append(Line2D([], [], marker="o", ls="", mfc="#cccccc", mec="#111", mew=0.4,
                                      ms=math.sqrt(sizes(np.array([max(mm, args.minmag)]))[0]),
                                      label=f"M{max(mm, args.minmag):g}"))
    axm.legend(handles=handles, loc="best", fontsize=7.8, framealpha=0.92, borderpad=0.6)

    # ---- colourbars under the map
    cb_y = max(0.065, map_bottom - 0.075)
    cbw = 0.42 * w
    cax1 = fig.add_axes([map_left + 0.04 * w, cb_y, cbw, 0.014])
    cb1 = fig.colorbar(mappable_slip, cax=cax1, orientation="horizontal")
    cb1.set_label("Slip (m)", fontsize=8.5)
    cb1.ax.tick_params(labelsize=7.5)
    if sc is not None:
        cax2 = fig.add_axes([map_left + 0.54 * w, cb_y, cbw, 0.014])
        cb2 = fig.colorbar(sc, cax=cax2, orientation="horizontal")
        cb2.set_label("Aftershock depth (km)" if args.color_by == "depth" else "Days after mainshock", fontsize=8.5)
        cb2.ax.tick_params(labelsize=7.5)

    # ---- fault-plane view(s)
    nseg = min(len(model.segments), 3)
    segs_sorted = sorted(model.segments, key=lambda s: -float(np.nansum(s.moment)) if s.moment is not None else 0)[:nseg]
    top, height = 0.875, 0.30
    each = height / nseg
    rx = min(0.615, max(0.45, map_left + w + 0.075))          # right column starts next to the map
    rw = 0.97 - rx
    fig.text(rx, 0.888, "Fault-plane view from the hanging wall (strike direction → right).\nArrows: hanging-wall slip "
             "direction, length ∝ slip; dashed lines: rupture-front time.", fontsize=7.6, color="#333", va="bottom")
    for k, s in enumerate(segs_sorted):
        ax = fig.add_axes([rx, top - (k + 1) * each + 0.04, rw, each - 0.065])
        _plane_panel(ax, s, frame, norm, hyp_ffm, nseg > 1, len(model.segments))

    # ---- cross-section
    axs = fig.add_axes([rx, 0.33, rw, 0.20])
    _section_panel(axs, model, frame, norm, main_pc, hyp_ffm, catalog if cat_ok else None, half_len, args, pmax)

    # ---- moment-rate function
    axr = fig.add_axes([rx, 0.075, 0.45 * rw, 0.17])
    _stf_panel(axr, stf, model)

    # ---- info box
    fig.text(rx + 0.52 * rw, 0.245, _info_text(ev, mt, ffp, model, catalog, checks, args), fontsize=7.6, va="top",
             ha="left", family="DejaVu Sans", linespacing=1.38)

    # ---- titles and footer
    t_utc = ev.time.strftime("%Y-%m-%d %H:%M:%S UTC")
    fig.text(0.035, 0.955, f"M{ev.mag:.1f} {ev.place}", fontsize=19, fontweight="bold")
    fig.text(0.035, 0.925, f"{t_utc}  ·  ComCat {ev.id}  ·  USGS finite-fault model "
             f"{ffp.code if ffp else ''} (updated {ffp.updated if ffp else 'n/a'})", fontsize=10.5, color="#444")
    foot = ("Data: USGS NEIC ComCat (origin, W-phase moment tensor, finite-fault model, FDSN event search). "
            "Finite-fault method: Ji et al. (2002), Hayes (2017). Beachball: lower-hemisphere equal-area projection. "
            "Coastlines: Natural Earth. Plate boundaries: Bird (2003), via github.com/fraxen/tectonicplates (ODC-By).\n"
            f"Figure: ffviz {__version__}, {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
            "Slip and geometry are plotted exactly as published by USGS; the finite-fault model is itself a "
            "non-unique inversion result and is revised as data improve.")
    fig.text(0.035, 0.012, foot, fontsize=7.2, color="#666", linespacing=1.45)

    fig.savefig(out_path, dpi=args.dpi, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------
def _plane_panel(ax, s, frame, norm, hyp, multi, nseg_total):
    # reference: projection of the hypocentre onto strike gives along = 0
    hx, hy = frame.to_xy(hyp[0], hyp[1])
    pc = plane_coords(s, frame, ref_xy=(float(hx), float(hy)))
    dx, dy = s.dx_km, s.dy_km
    rects = [[(a - dx / 2, w - dy / 2), (a + dx / 2, w - dy / 2), (a + dx / 2, w + dy / 2), (a - dx / 2, w + dy / 2)]
             for a, w in zip(pc.along, pc.downdip)]
    ax.add_collection(PolyCollection(rects, array=s.slip, cmap=SLIP_CMAP, norm=norm, edgecolors=(0, 0, 0, 0.08),
                                     linewidths=0.2))
    # rupture-time contours
    if s.t_rup is not None and np.ptp(s.t_rup) > 1 and s.n >= 6:
        step = _nice(np.ptp(s.t_rup) / 5.0)
        levels = np.arange(step, np.nanmax(s.t_rup) + step, step)
        try:
            cs = ax.tricontour(pc.along, pc.downdip, s.t_rup, levels=levels, colors="#555", linewidths=0.7,
                               linestyles="--")
            ax.clabel(cs, cs.levels[::2], fmt=lambda v: f"{v:g} s", fontsize=7)
        except Exception:
            pass
    # rake arrows (hanging-wall slip direction), length ∝ slip
    big = s.slip >= 0.15 * norm.vmax
    if np.any(big):
        r = np.radians(s.rake[big])
        f = s.slip[big] / norm.vmax
        ax.quiver(pc.along[big], pc.downdip[big], np.cos(r) * f, -np.sin(r) * f, angles="xy", scale_units="xy",
                  scale=1.0 / (0.95 * min(dx, dy)), width=0.0045, color="#2b2b2b", zorder=3)
    # hypocentre if it lies on this segment
    sd = math.sin(math.radians(pc.dip))
    w_h = (hyp[2] - pc.top_depth) / sd if hyp[2] == hyp[2] else None
    a_h = 0.0
    inside = w_h is not None and (pc.along.min() - dx <= a_h <= pc.along.max() + dx) and \
        (-dy <= w_h <= pc.width + dy)
    if inside:
        ax.scatter([a_h], [w_h], marker="*", s=260, c="white", edgecolors="#111", linewidths=1.1, zorder=5)
    ax.set_xlim(pc.along.min() - dx, pc.along.max() + dx)
    ax.set_ylim(pc.width + dy * 0.5, -dy * 0.5)
    ax.set_aspect("equal")
    lab = f"Segment {s.index}: " if nseg_total > 1 else "Fault plane: "
    ax.set_title(f"{lab}strike {pc.strike:.0f}°, dip {pc.dip:.0f}°", fontsize=8.6, loc="left")
    ax.set_xlabel("Along strike (km) →", fontsize=8)
    ax.set_ylabel("Down dip (km)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    sec = ax.secondary_yaxis("right", functions=(lambda w: pc.top_depth + w * sd,
                                                  lambda z: (z - pc.top_depth) / sd))
    sec.set_ylabel("Depth (km)", fontsize=8)
    sec.tick_params(labelsize=7.5)


def _section_panel(ax, model, frame, norm, main_pc, hyp, catalog, half_len, args, pmax):
    hx, hy = frame.to_xy(hyp[0], hyp[1])
    strike = main_pc.strike
    from matplotlib.collections import LineCollection
    segs_xy, cols, P, Z, Qall = [], [], [], [], []
    for s in model.segments:
        p, q = profile_coords(frame, s.lon, s.lat, strike)
        p = p - _pp(hx, hy, strike)
        Qall.append(q - _qq(hx, hy, strike))
        pc = plane_coords(s, frame)
        row = np.round(pc.downdip / s.dy_km).astype(int)
        dip = math.radians(pc.dip)
        # the down-dip direction projected on the section (accounts for segments with other strikes)
        cosang = math.cos(math.radians(pc.strike - strike))
        hp, hz = 0.5 * s.dy_km * math.cos(dip) * cosang, 0.5 * s.dy_km * math.sin(dip)
        for r in np.unique(row):
            m = row == r
            pm, zm = float(np.median(p[m])), float(np.median(s.depth[m]))
            segs_xy.append([(pm - hp, zm - hz), (pm + hp, zm + hz)])
            cols.append(float(np.nanmax(s.slip[m])))
            P.extend([pm - hp, pm + hp])
            Z.extend([zm - hz, zm + hz])
    lc = LineCollection(segs_xy, array=np.array(cols), cmap=SLIP_CMAP, norm=norm, linewidths=6, zorder=3,
                        capstyle="butt")
    ax.add_collection(lc)
    P, Z = np.array(P), np.array(Z)
    Qall = np.concatenate(Qall)
    q_lo, q_hi = float(Qall.min()) - 20.0, float(Qall.max()) + 20.0      # model extent along strike + 20 km
    xmin, xmax = min(P.min(), -0.5 * pmax) - 10, max(P.max(), pmax) + 10
    n_sel = 0
    if catalog is not None:
        p, q = profile_coords(frame, catalog.lon, catalog.lat, strike)
        p = p - _pp(hx, hy, strike)
        q = q - _qq(hx, hy, strike)
        sel = (q >= q_lo) & (q <= q_hi) & ~catalog.fixed_depth & (catalog.time >= 0) & (p >= xmin) & (p <= xmax)
        n_sel = int(sel.sum())
        ax.scatter(p[sel], catalog.depth[sel], s=6 + 9 * np.clip(catalog.mag[sel] - args.minmag + 0.3, 0.1, None) ** 2.2,
                   facecolors="none", edgecolors="#1d3557", linewidths=0.7, alpha=0.8, zorder=4)
    ax.scatter([0], [hyp[2]], marker="*", s=230, c="white", edgecolors="#111", linewidths=1.1, zorder=5)
    zmax = max(float(np.nanmax(Z)) + 15, hyp[2] + 15)
    if catalog is not None and n_sel:
        zmax = max(zmax, float(np.nanpercentile(catalog.depth[sel], 98)) + 10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmax, -3)
    ax.axhline(0, color="#56626b", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Distance along A–A′ (N{(strike + 90) % 360:.0f}°E) from hypocentre (km)", fontsize=8)
    ax.set_ylabel("Depth (km)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.set_title(f"Section A–A′ · fault coloured by max. slip along strike per down-dip row\n"
                 f"○ aftershocks along the rupture length (+20 km), non-default depths only (n={n_sel})",
                 fontsize=8.0, loc="left")
    ax.text(0.01, 0.97, "A", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left", color="#1d3557")
    ax.text(0.99, 0.97, "A′", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="right", color="#1d3557")


def _pp(x, y, strike):
    phi = math.radians(strike)
    return float(x) * math.cos(phi) - float(y) * math.sin(phi)


def _qq(x, y, strike):
    phi = math.radians(strike)
    return float(x) * math.sin(phi) + float(y) * math.cos(phi)


def _stf_panel(ax, stf, model):
    ax.set_title("Moment-rate function", fontsize=8.6, loc="left")
    if stf is None:
        ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#777")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    t, r = stf
    sc = 10 ** math.floor(math.log10(max(np.nanmax(r), 1.0)))
    ax.fill_between(t, r / sc, color="#bbbbbb", alpha=0.7)
    ax.plot(t, r / sc, color="#222", lw=1.0)
    cum = np.concatenate([[0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(t))])
    m0 = cum[-1]
    t95 = float(np.interp(0.95 * m0, cum, t)) if m0 > 0 else float("nan")
    ax.axvline(t95, color="#b3121a", lw=0.9, ls="--")
    ax.text(0.97, 0.95, f"M$_0$ = {m0:.2e} N·m\nM$_w$ = {mw_from_m0(m0):.2f}\nT$_{{95}}$ = {t95:.0f} s",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    ax.set_xlim(0, float(t.max()))
    ax.set_ylim(0, None)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel(f"Moment rate (10$^{{{int(math.log10(sc))}}}$ N·m/s)", fontsize=8)
    ax.tick_params(labelsize=7.5)


def _info_text(ev, mt, ffp, model, catalog, checks, args):
    from .checks import summarize
    L = []
    L.append(r"$\bf{Hypocentre}$ (ComCat)")
    L.append(f"{ev.lat:.3f}°, {ev.lon:.3f}°, {ev.depth:.1f} km")
    if mt is not None:
        L.append(r"$\bf{Moment\ tensor}$" + f" ({mt.source.upper()} {mt.method})")
        if mt.mw:
            L.append(f"M$_w$ {mt.mw:.2f}" + (f", M$_0$ {mt.m0:.2e} N·m" if mt.m0 else ""))
        if mt.np1 and mt.np2:
            L.append(f"NP1 {mt.np1[0]:.0f}/{mt.np1[1]:.0f}/{mt.np1[2]:.0f}  NP2 {mt.np2[0]:.0f}/{mt.np2[1]:.0f}/{mt.np2[2]:.0f}")
        extra = []
        if mt.depth is not None:
            extra.append(f"centroid {mt.depth:.1f} km")
        if mt.pct_dc is not None:
            extra.append(f"DC {mt.pct_dc:.0f}%")
        if extra:
            L.append(", ".join(extra))
    L.append(r"$\bf{Finite\ fault}$" + f" ({len(model.segments)} segment{'s' if len(model.segments) > 1 else ''})")
    if len(model.segments) > 3:
        L.append(f"(3 largest of {len(model.segments)} segments shown)")
    for s in model.segments[:3]:
        L.append(f"seg {s.index}: {s.mean_strike:.0f}°/{s.mean_dip:.0f}°, {s.nx or '?'}×{s.ny or '?'} × "
                 f"{s.dx_km:g}×{s.dy_km:g} km")
    m0 = model.total_moment
    if m0:
        L.append(f"ΣM$_0$ {m0:.2e} N·m (M$_w$ {mw_from_m0(m0):.2f})")
    L.append(f"max slip {model.max_slip:.2f} m")
    if ffp is not None:
        vr = ffp.prop("average-rupture-velocity")
        if vr:
            L.append(f"mean rupture velocity {vr:.2f} km/s")
    if catalog is not None:
        q = catalog.query
        L.append(r"$\bf{Aftershocks}$" + f" (ComCat, M ≥ {float(q['minmagnitude']):g})")
        L.append(f"{len(catalog)} events, r ≤ {q['maxradiuskm']} km")
        L.append(f"{q['starttime'][:10]} – {q['endtime'][:10]}")
    L.append(r"$\bf{Consistency\ checks}$: " + summarize(checks))
    return "\n".join(L)
