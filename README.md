# ffviz – earthquake rupture figures from USGS data

**ffviz** turns the published USGS finite-fault model of a large earthquake into
one clear figure: where the fault slipped, how the rupture spread, the focal
mechanism, the aftershocks and the moment-rate function – all on a single page.

<img width="3500" height="2200" alt="us6000u18k_ffm" src="https://github.com/user-attachments/assets/bee73002-1c5e-493d-b1d5-1871157a191c" />



## Download and use (Windows 10/11)

1. Open **[Releases](../../releases)** and download **`ffviz-windows.zip`** (under *Assets*).
2. Right-click the zip → **Extract All…**
3. Double-click **`ffviz.exe`** in the extracted folder.
4. If Windows says *“Windows protected your PC”*: click **More info → Run anyway**
   (the app is not code-signed; all source code is in this repository).

No installation and no Python needed. An internet connection is required.

## How it works

1. Pick an earthquake from the list (all recent USGS events with a finite-fault model)
   or type an event ID, e.g. `us6000u18k`. The ID is in the address of the USGS event page.
2. Click **Create figure**.
3. The figure appears in the window and is saved to `Documents\ffviz` (PNG, PDF or SVG).

When USGS updates a model, ffviz picks up the new version automatically the next
time you create the figure.

## What the figure shows

| Panel | Content |
|---|---|
| Map | slip on the fault, upper fault edge, epicentre, beachball, aftershocks, plate boundaries |
| Fault plane | slip seen from the hanging wall, slip direction, rupture-front times |
| Cross-section | fault and aftershocks in depth |
| Moment rate | how fast the earthquake released its energy |
| Checks | automatic comparison of the plotted model with the values published by USGS |

All values come directly from the USGS files – nothing is estimated or redrawn by hand.
A finite-fault model is a scientific model, not a direct measurement, and USGS
revises it as more data arrive.

## Privacy

No account, no tracking, no personal data. ffviz only downloads public data from
USGS, Natural Earth (coastlines) and GitHub (plate boundaries).

## Data sources and credits

* USGS NEIC ComCat – earthquakes, moment tensors, finite-fault models (public domain)
* Ji, Wald & Helmberger (2002), BSSA 92(4) and Hayes (2017), EPSL 468 – finite-fault method
* Bird (2003), G³ 4(3):1027 – plate boundaries (GeoJSON: github.com/fraxen/tectonicplates, ODC-By 1.0)
* Natural Earth – coastlines and borders (public domain)

Licence: MIT (see `LICENSE`). Bundled third-party software: see `THIRD_PARTY_NOTICES.txt`.

---

<details>
<summary><b>For developers</b> – run from source, command line, building the exe, tests</summary>

### Run from source (Python 3.10–3.13)

Double-click `install.bat` once, then `Start ffviz.bat` – or manually:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m ffviz.gui                 # desktop app
python -m ffviz us6000u18k          # command line
```

### Command-line options

| option | meaning |
|---|---|
| `--out file.png/.pdf/.svg` | output file |
| `--days`, `--days-before` | aftershock window (default 30 days after) |
| `--minmag` | minimum aftershock magnitude (default 4.5; 2.5 for Mw < 6) |
| `--radius` | search radius in km (default from fault size) |
| `--color-by time\|depth` | aftershock colouring |
| `--coast-res 10m\|50m\|110m` | coastline resolution |
| `--no-coast`, `--no-plates`, `--no-aftershocks` | switch layers off |
| `--offline` | use cached files only |
| `--refresh` | download all files again |
| `--strict` | exit with error if a consistency check fails |

Downloaded files are cached per product version in `%LOCALAPPDATA%\ffviz`.

### Consistency checks

Subfault grid complete · grid spacing · outline vs. published corners · implied
rigidity (also decides cm/m and dyne·cm/N·m) · Σ subfault moments vs. USGS scalar
moment · Mw (finite fault) vs. Mw (moment tensor) · maximum slip vs. USGS ·
hypocentre on the fault plane · fault plane vs. moment-tensor nodal plane.

### Conventions and limitations

* Strike/dip/rake after Aki & Richards; Mw = 2/3 (log10 M0 − 9.1).
* Beachballs: lower-hemisphere equal-area projection, computed from the full moment tensor.
* Distances in a local azimuthal-equidistant projection (WGS84); dateline-safe.
* Aftershock depths of exactly 0, 5, 10, 33 or 35 km are treated as probable fixed
  depths and left out of the cross-section (heuristic – ComCat does not flag them).
* The cross-section projects all segments onto one plane; for curved multi-segment
  ruptures use the fault-plane panels.

### Build the Windows program

```powershell
install.bat
build_exe.bat      # -> dist\ffviz\ffviz.exe and dist\ffviz-windows.zip
```

### Tests

```powershell
pip install pytest
python -m pytest tests
```

The tests create synthetic events in the USGS file formats and check parsing,
units, geometry, moment, magnitude, beachballs (against P/T axes and ObsPy if
installed), caching/updates, dateline handling and the command line.

</details>
