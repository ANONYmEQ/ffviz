import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixtures import write_event, write_catalog                        # noqa: E402
from ffviz.ffm import read_param, read_fsp, resolve_units, read_moment_rate, mw_from_m0, FFMFormatError  # noqa: E402
from ffviz.geometry import LocalFrame, plane_coords, segment_outline, wrap_lon, subfault_corners  # noqa: E402
from ffviz.usgs import load_event, moment_tensor, load_catalog            # noqa: E402
from ffviz.checks import run_checks, FAIL, WARN                          # noqa: E402


@pytest.fixture(scope="module", params=["panama", "two_segment", "dateline"])
def event(request, tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("cache"))
    eid, info = write_event(cache, request.param)
    return cache, eid, info


def _files(cache, eid):
    d = os.path.join(cache, eid)
    sub = [x for x in os.listdir(d) if x.startswith("finite-fault")][0]
    return os.path.join(d, sub)


def test_mw_formula():
    # Hanks & Kanamori: M0 = 10^(1.5 Mw + 9.1)
    for mw in (5.0, 7.65, 9.1):
        assert mw_from_m0(10 ** (1.5 * mw + 9.1)) == pytest.approx(mw, abs=1e-12)


def test_param_units_and_moment(event):
    cache, eid, info = event
    txt = open(os.path.join(_files(cache, eid), "basic_inversion.param")).read()
    m = read_param(txt)
    resolve_units(m)                                   # without reference values
    assert m.slip_unit == "cm" and m.moment_unit == "dyne-cm"
    assert m.max_slip == pytest.approx(info["max_slip"], rel=1e-3)
    assert m.total_moment == pytest.approx(info["m0"], rel=1e-3)
    assert 25e9 < m.rigidity_pa < 35e9
    for s in m.segments:
        assert s.n == s.nx * s.ny


def test_fsp_matches_param(event):
    cache, eid, info = event
    d = _files(cache, eid)
    p = resolve_units(read_param(open(os.path.join(d, "basic_inversion.param")).read()))
    f = resolve_units(read_fsp(open(os.path.join(d, "complete_inversion.fsp")).read()))
    assert f.slip_unit == "m" and f.moment_unit == "N-m"
    assert len(f.segments) == len(p.segments)
    assert f.total_moment == pytest.approx(p.total_moment, rel=1e-3)
    assert f.max_slip == pytest.approx(p.max_slip, rel=1e-3)


def test_geometry_matches_published_outline(event):
    """ffviz (pyproj/WGS84) must reproduce the independently computed outline."""
    cache, eid, info = event
    m = resolve_units(read_param(open(os.path.join(_files(cache, eid), "basic_inversion.param")).read()))
    lat0, lon0, _ = info["hyp"]
    fr = LocalFrame(lat0, lon0)
    for s in m.segments:
        ours = segment_outline(s, fr)
        for (blon, blat, bdep), (olon, olat, odep) in zip(s.boundary[:4], ours):
            bx, by = fr.to_xy(blon, blat)
            ox, oy = fr.to_xy(olon, olat)
            assert math.sqrt((bx - ox) ** 2 + (by - oy) ** 2 + (bdep - odep) ** 2) < 0.3


def test_plane_coords_spacing(event):
    cache, eid, info = event
    m = resolve_units(read_param(open(os.path.join(_files(cache, eid), "basic_inversion.param")).read()))
    fr = LocalFrame(*info["hyp"][:2])
    for s in m.segments:
        pc = plane_coords(s, fr)
        assert pc.length == pytest.approx(s.nx * s.dx_km, rel=0.03)
        assert pc.width == pytest.approx(s.ny * s.dy_km, rel=0.01)
        assert pc.top_depth >= -0.5


def test_corner_order_is_top_first(event):
    cache, eid, info = event
    m = resolve_units(read_param(open(os.path.join(_files(cache, eid), "basic_inversion.param")).read()))
    fr = LocalFrame(*info["hyp"][:2])
    lon, lat, dep = subfault_corners(m.segments[0], fr)
    assert np.all(dep[:, 0] < dep[:, 3]) and np.all(dep[:, 1] < dep[:, 2])


def test_wrap_lon():
    assert wrap_lon(-179.9, 179.9) == pytest.approx(180.1)
    assert wrap_lon(179.9, -179.9) == pytest.approx(-180.1)


def test_moment_rate_unit_detection(event):
    cache, eid, info = event
    txt = open(os.path.join(_files(cache, eid), "moment_rate.mr")).read()
    t, r, note = read_moment_rate(txt, ref_moment_nm=info["m0"])
    integ = np.trapezoid(r, t) if hasattr(np, "trapezoid") else np.trapz(r, t)
    assert integ == pytest.approx(info["m0"], rel=0.01)


def test_checks_all_pass(event):
    cache, eid, info = event
    ev = load_event(eid, cache, offline=True)
    ffp = ev.preferred("finite-fault")
    m = resolve_units(read_param(open(os.path.join(_files(cache, eid), "basic_inversion.param")).read()),
                      ref_moment_nm=ffp.prop("scalar-moment"), ref_max_slip_m=ffp.prop("maximum-slip"))
    mt = moment_tensor(ev)
    write_catalog(cache, eid, info, 250, 4.5)
    cat = load_catalog(ev, cache, 250, 0.0, 30.0, 4.5, offline=True)
    checks = run_checks(m, LocalFrame(ev.lat, ev.lon), ffp.properties, mt, ev, cat)
    assert not [c for c in checks if c.status == FAIL], checks
    assert len(cat) == 60                     # mainshock removed
    assert cat.fixed_depth.sum() > 0


def test_moment_tensor_parsing(event):
    cache, eid, info = event
    mt = moment_tensor(load_event(eid, cache, offline=True))
    assert mt.np1 and mt.np2 and len(mt.tensor) == 6
    assert mt.pct_dc == pytest.approx(78.0)
    # scalar moment of the tensor (Silver & Jordan norm) equals M0
    mrr, mtt, mpp, mrt, mrp, mtp = mt.tensor
    M = np.array([[mrr, mrt, mrp], [mrt, mtt, mtp], [mrp, mtp, mpp]])
    assert math.sqrt(0.5 * np.sum(M * M)) == pytest.approx(info["m0"], rel=1e-3)


def test_bad_file_raises():
    with pytest.raises(FFMFormatError):
        read_param("# nothing here\n1 2 3\n")


def test_cli_end_to_end(event, tmp_path):
    cache, eid, info = event
    from ffviz.__main__ import main
    write_catalog(cache, eid, info, 250, 4.5)
    out = str(tmp_path / f"{eid}.png")
    rc = main([eid, "--offline", "--cache-dir", cache, "--radius", "250", "--no-coast", "--no-plates",
               "--dpi", "60", "--out", out, "--strict"])
    assert rc == 0 and os.path.getsize(out) > 10_000


# ------------------------------------------------------------------ beachball
from ffviz.beachball import aux_plane, pt_axes, project, radiation_grid, sdr_to_tensor, tensor_ned  # noqa: E402


@pytest.mark.parametrize("np1,np2", [((288, 66, -17), (25, 75, -155)),            # USGS us6000u18k
                                     ((160.6, 44.59, 76.78), (358.86, 46.88, 102.71))])  # USGS us7000srb1
def test_aux_plane_matches_usgs(np1, np2):
    s, d, r = aux_plane(*np1)
    assert min(abs(s - np2[0]), 360 - abs(s - np2[0])) < 1.0
    assert d == pytest.approx(np2[1], abs=1.0) and r == pytest.approx(np2[2], abs=1.0)


def test_beachball_polarity_and_projection():
    rng = np.random.default_rng(3)
    for _ in range(200):
        sdr = (rng.uniform(0, 360), rng.uniform(5, 89), rng.uniform(-179, 179))
        (pt, pp), (tt, tp) = pt_axes(*sdr)
        X, Y, A = radiation_grid(sdr_to_tensor(*sdr), n=401)
        for (tr, pl), sign in (((tt, tp), 1), ((pt, pp), -1)):
            x, y = project(tr, pl)
            j, i = int(round((x + 1) / 2 * 400)), int(round((y + 1) / 2 * 400))
            assert np.sign(A[i, j]) == sign


def test_tensor_is_double_couple_and_traceless():
    M = tensor_ned(sdr_to_tensor(123, 45, 67, 1.0))
    w = np.sort(np.linalg.eigvalsh(M))
    assert np.trace(M) == pytest.approx(0, abs=1e-12)
    assert w == pytest.approx([-1, 0, 1], abs=1e-12)


def test_beachball_against_obspy_if_installed():
    beach = pytest.importorskip("obspy.imaging.beachball").beach
    rng = np.random.default_rng(5)
    for _ in range(60):
        sdr = (rng.uniform(0, 360), rng.uniform(5, 89), rng.uniform(-179, 179))
        M = tensor_ned(sdr_to_tensor(*sdr))
        col = beach(list(sdr), xy=(0, 0), width=2, linewidth=0)
        paths, fcs = col.get_paths(), col.get_facecolors()
        for _ in range(30):
            x, y = rng.uniform(-0.9, 0.9, 2)
            if x * x + y * y > 0.8:
                continue
            i = 2 * math.asin(math.hypot(x, y) * math.sin(math.pi / 4))
            az = math.atan2(x, y)
            g = np.array([math.sin(i) * math.cos(az), math.sin(i) * math.sin(az), math.cos(i)])
            a = g @ M @ g
            if abs(a) < 0.15:          # close to a nodal line: polygon approximations differ
                continue
            filled = any(p.contains_point((x, y)) and not np.allclose(fc[:3], 1.0) for p, fc in zip(paths, fcs))
            assert filled == (a > 0)


def test_invalid_event_id_rejected(tmp_path):
    from ffviz.__main__ import main
    assert main(["../etc", "--offline", "--cache-dir", str(tmp_path)]) == 2


def test_event_refresh_falls_back_to_cache_and_detects_update(event, monkeypatch):
    import json as _json
    from ffviz import usgs
    cache, eid, info = event
    path = os.path.join(cache, eid, "event.geojson")
    original = open(path).read()
    # 1) no network: cached copy is used
    def fail(*a, **k):
        raise usgs.DataError("no network")
    monkeypatch.setattr(usgs, "_http_get", fail)
    ev = usgs.load_event(eid, cache)
    assert ev.id == eid and not ev.notes
    # 2) USGS publishes a new finite-fault version -> picked up and reported
    det = _json.loads(original)
    ff = det["properties"]["products"]["finite-fault"][0]
    ff["code"], ff["updateTime"] = eid + "_2", ff["updateTime"] + 3600_000
    monkeypatch.setattr(usgs, "_http_get", lambda *a, **k: _json.dumps(det).encode())
    ev = usgs.load_event(eid, cache)
    assert ev.preferred("finite-fault").code == eid + "_2" and ev.notes
    with open(path, "w") as f:      # restore fixture for the other tests
        f.write(original)
