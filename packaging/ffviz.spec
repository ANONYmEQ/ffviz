# PyInstaller spec for the ffviz desktop program.
# Build with:  pyinstaller packaging/ffviz.spec --noconfirm   (see build_exe.bat)
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

here = os.path.abspath(SPECPATH)
root = os.path.dirname(here)

datas = []
datas += collect_data_files("cartopy")       # cartopy's own small data files
datas += collect_data_files("pyproj")        # PROJ database (proj.db)
datas += collect_data_files("certifi")       # HTTPS certificates
datas += [(os.path.join(root, "ffviz", "icon.png"), "ffviz")]
hidden = collect_submodules("ffviz") + ["PIL._tkinter_finder", "shapely", "pyproj.database"]

a = Analysis(
    [os.path.join(here, "ffviz_app.py")],
    pathex=[root],
    datas=datas,
    hiddenimports=hidden,
    excludes=["obspy", "scipy", "IPython", "pytest", "sphinx", "PyQt5", "PySide2", "PyQt6", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ffviz",
    console=False,                # windowed app, no black console
    icon=os.path.join(here, "ffviz.ico"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="ffviz")
