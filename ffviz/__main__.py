"""Command-line interface.

Examples
--------
    python -m ffviz us6000u18k
    python -m ffviz us7000srb1 --days 30 --minmag 5 --out mindanao.pdf
    python -m ffviz us6000u18k --offline          # re-draw from the cache only
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

from . import __version__
from .checks import run_checks, summarize, FAIL
from .ffm import read_model, resolve_units, read_moment_rate, FFMFormatError
from .geometry import LocalFrame, plane_coords
from .usgs import load_event, load_product_file, moment_tensor, load_catalog, DataError


def default_cache() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "ffviz")


def build_parser():
    p = argparse.ArgumentParser(prog="ffviz", description="Publication-style figure of a USGS finite-fault model.")
    p.add_argument("eventid", help="ComCat event id, e.g. us6000u18k")
    p.add_argument("-o", "--out", help="output file (.png, .pdf, .svg); default <eventid>_ffm.png")
    p.add_argument("--days", type=float, default=30.0, help="aftershock window after the mainshock (days, default 30)")
    p.add_argument("--days-before", type=float, default=0.0, help="include foreshocks this many days before")
    p.add_argument("--minmag", type=float, default=None, help="minimum aftershock magnitude (default: 4.5, or 2.5 for Mw<6)")
    p.add_argument("--radius", type=float, default=None, help="aftershock search radius in km (default: from fault size)")
    p.add_argument("--color-by", choices=("time", "depth"), default="time", help="aftershock colour")
    p.add_argument("--coast-res", choices=("10m", "50m", "110m"), default=None, help="Natural Earth resolution")
    p.add_argument("--no-coast", action="store_true", help="do not draw coastlines (no download needed)")
    p.add_argument("--no-plates", action="store_true", help="do not draw plate boundaries")
    p.add_argument("--no-aftershocks", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--cache-dir", default=default_cache(), help="where downloaded files are stored")
    p.add_argument("--offline", action="store_true", help="use cached files only")
    p.add_argument("--refresh", action="store_true", help="download all files again (normally only new versions are fetched)")
    p.add_argument("--strict", action="store_true", help="exit with error if a consistency check fails")
    p.add_argument("--version", action="version", version=f"ffviz {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_\-]{2,40}", args.eventid):
        print(f"error: '{args.eventid}' is not a valid ComCat event id (e.g. us6000u18k)", file=sys.stderr)
        return 2
    kw = dict(offline=args.offline, refresh=args.refresh)
    try:
        ev = load_event(args.eventid, args.cache_dir, **kw)
    except DataError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{ev.title}  ({ev.time:%Y-%m-%d %H:%M:%S} UTC)")
    for note in ev.notes:
        print(f"note: {note}")

    ffp = ev.preferred("finite-fault")
    if ffp is None:
        print("error: USGS has not published a finite-fault model for this event.", file=sys.stderr)
        return 3
    fname = ffp.find_file(".param", ".fsp")
    if fname is None:
        print(f"error: finite-fault product has no .param/.fsp file (files: {list(ffp.contents)})", file=sys.stderr)
        return 3
    print(f"finite-fault product {ffp.code} ({ffp.source}), updated {ffp.updated}; file {fname}")
    try:
        model = read_model(load_product_file(ev, ffp, fname, args.cache_dir, **kw), fname)
    except FFMFormatError as e:
        alt = ffp.find_file(".fsp") if fname.lower().endswith(".param") else None
        if not alt:
            print(f"error: {e}", file=sys.stderr)
            return 4
        print(f"warning: {e}; trying {alt}")
        model = read_model(load_product_file(ev, ffp, alt, args.cache_dir, **kw), alt)
    resolve_units(model, ref_moment_nm=ffp.prop("scalar-moment"), ref_max_slip_m=ffp.prop("maximum-slip"))

    mt = moment_tensor(ev)

    stf = None
    mr = ffp.find_file(".mr")
    if mr:
        try:
            t, r, _ = read_moment_rate(load_product_file(ev, ffp, mr, args.cache_dir, **kw),
                                       ref_moment_nm=model.total_moment or ffp.prop("scalar-moment"))
            stf = (t, r)
        except (FFMFormatError, DataError) as e:
            print(f"warning: moment-rate function not used ({e})")

    frame = LocalFrame(ev.lat, ev.lon)
    mw = (mt.mw if mt and mt.mw else ev.mag)
    if args.minmag is None:
        args.minmag = 4.5 if mw >= 6.0 else 2.5
    if args.radius is None:
        size = max(plane_coords(s, frame).length for s in model.segments)
        args.radius = max(100.0, 1.5 * size + 50.0)

    catalog = None
    if not args.no_aftershocks:
        try:
            catalog = load_catalog(ev, args.cache_dir, args.radius, args.days_before, args.days, args.minmag, **kw)
            print(f"aftershocks: {len(catalog)} events (M>={args.minmag:g}, r<={args.radius:.0f} km)")
        except DataError as e:
            print(f"warning: no aftershock catalogue ({e})")

    checks = run_checks(model, frame, ffp.properties, mt, ev, catalog)
    print("\nconsistency checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    print(f"  -> {summarize(checks)}")
    if args.strict and any(c.status == FAIL for c in checks):
        print("error: a consistency check failed (--strict)", file=sys.stderr)
        return 5

    out = args.out or f"{args.eventid}_ffm.png"
    from .plot import make_figure
    try:
        make_figure(ev, mt, ffp, model, catalog, stf, checks, args, out)
    except PermissionError:
        print(f"error: cannot write {out} - is the file open in another program?", file=sys.stderr)
        return 6
    print(f"\nfigure written to {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
