"""Desktop app for ffviz (Tkinter, ships with Python - no extra install).

Start with   pythonw -m ffviz.gui   or double-click "Start ffviz.bat" / ffviz.exe.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")            # drawing happens in a worker thread -> non-GUI backend

import io
import json
import os
import queue
import sys
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from . import __version__
from .__main__ import main as cli_main, default_cache

FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def default_outdir() -> str:
    """Documents folder (also when Windows redirects it to OneDrive)."""
    home = os.path.expanduser("~")
    cands = [os.path.join(home, "Documents")]
    od = os.environ.get("OneDrive")
    if od:
        cands = [os.path.join(od, "Documents"), os.path.join(od, "Dokumente")] + cands
    base = next((c for c in cands if os.path.isdir(c)), home)
    return os.path.join(base, "ffviz")


class _QueueWriter(io.TextIOBase):
    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(("log", s))
        return len(s)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"ffviz {__version__} – finite-fault figures from USGS data")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self.last_file = None
        self._img = None
        self._img_src = None
        try:
            icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
            self._icon = tk.PhotoImage(file=icon)
            self.iconphoto(True, self._icon)
        except Exception:
            pass
        self._build()
        self.after(100, self._poll)
        self.load_events()

    # ------------------------------------------------------------------ layout
    def _build(self):
        style = ttk.Style(self)
        try:
            style.theme_use("vista" if sys.platform == "win32" else "clam")
        except tk.TclError:
            pass
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=8)

        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(pw, width=470)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        # --- event list
        box = ttk.LabelFrame(left, text=" 1. Select an earthquake (USGS events with a finite-fault model) ")
        box.pack(fill="both", expand=True)
        top = ttk.Frame(box)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="M ≥").pack(side="left")
        self.list_mag = tk.StringVar(value="6.5")
        ttk.Spinbox(top, from_=5.0, to=9.0, increment=0.5, width=5, textvariable=self.list_mag).pack(side="left", padx=(2, 10))
        ttk.Label(top, text="last").pack(side="left")
        self.list_years = tk.StringVar(value="2")
        ttk.Spinbox(top, from_=1, to=30, width=4, textvariable=self.list_years).pack(side="left", padx=2)
        ttk.Label(top, text="years").pack(side="left")
        ttk.Button(top, text="Load list", command=self.load_events).pack(side="right")

        cols = ("date", "mag", "place")
        self.tree = ttk.Treeview(box, columns=cols, show="headings", height=14, selectmode="browse")
        for c, t, w in (("date", "Date (UTC)", 115), ("mag", "M", 45), ("place", "Location", 280)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w" if c == "place" else "center", stretch=(c == "place"))
        sb = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=(0, 6))
        sb.pack(side="left", fill="y", pady=(0, 6), padx=(0, 6))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda e: self.run())

        # --- id + options
        opt = ttk.LabelFrame(left, text=" 2. Options ")
        opt.pack(fill="x", pady=(8, 0))
        g = ttk.Frame(opt)
        g.pack(fill="x", padx=6, pady=6)
        self.eid = tk.StringVar()
        self.days = tk.StringVar(value="30")
        self.minmag = tk.StringVar(value="")
        self.color = tk.StringVar(value="time")
        self.fmt = tk.StringVar(value="PNG")
        self.plates = tk.BooleanVar(value=True)
        self.coast = tk.BooleanVar(value=True)
        self.refresh = tk.BooleanVar(value=False)
        self.outdir = tk.StringVar(value=default_outdir())

        def row(r, label, widget):
            ttk.Label(g, text=label).grid(row=r, column=0, sticky="w", pady=2)
            widget.grid(row=r, column=1, sticky="we", pady=2, padx=(8, 0))
        g.columnconfigure(1, weight=1)
        row(0, "Event ID", ttk.Entry(g, textvariable=self.eid))
        row(1, "Aftershocks: days", ttk.Spinbox(g, from_=1, to=3650, width=8, textvariable=self.days))
        row(2, "Aftershocks: min. magnitude", ttk.Entry(g, textvariable=self.minmag, width=8))
        ttk.Label(g, text="(empty = automatic)", foreground="#777").grid(row=2, column=1, sticky="e")
        row(3, "Colour aftershocks by", ttk.Combobox(g, textvariable=self.color, values=("time", "depth"),
                                                     state="readonly", width=8))
        row(4, "File format", ttk.Combobox(g, textvariable=self.fmt, values=("PNG", "PDF", "SVG"),
                                           state="readonly", width=8))
        chk = ttk.Frame(g)
        chk.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(chk, text="Coastlines", variable=self.coast).pack(side="left")
        ttk.Checkbutton(chk, text="Plate boundaries", variable=self.plates).pack(side="left", padx=10)
        ttk.Checkbutton(chk, text="Re-download all files", variable=self.refresh).pack(side="left")
        of = ttk.Frame(g)
        of.grid(row=6, column=0, columnspan=2, sticky="we", pady=(6, 0))
        ttk.Label(of, text="Save to").pack(side="left")
        ttk.Entry(of, textvariable=self.outdir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(of, text="…", width=3, command=self._pick_dir).pack(side="left")

        self.go = ttk.Button(left, text="▶  Create figure", style="Big.TButton", command=self.run)
        self.go.pack(fill="x", pady=8)
        self.prog = ttk.Progressbar(left, mode="indeterminate")
        self.prog.pack(fill="x")

        # --- right: preview + log
        self.status = tk.StringVar(value="Select an earthquake on the left and click “Create figure”.")
        ttk.Label(right, textvariable=self.status, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(right, background="#f4f4f4", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(4, 4))
        self.canvas.bind("<Configure>", lambda e: self._show_image())
        btns = ttk.Frame(right)
        btns.pack(fill="x")
        self.b_open = ttk.Button(btns, text="Open file", command=self._open_file, state="disabled")
        self.b_open.pack(side="left")
        self.b_dir = ttk.Button(btns, text="Open folder", command=self._open_dir)
        self.b_dir.pack(side="left", padx=6)
        self.log = tk.Text(right, height=11, font=("Consolas", 9), wrap="word", background="#111", foreground="#ddd")
        self.log.pack(fill="x", pady=(6, 0))
        for tag, col in (("PASS", "#7bd88f"), ("WARN", "#ffd166"), ("FAIL", "#ff6b6b"), ("err", "#ff6b6b")):
            self.log.tag_configure(tag, foreground=col)

    # ------------------------------------------------------------------ events list
    def load_events(self):
        try:
            mag = float(self.list_mag.get().replace(",", "."))
            years = float(self.list_years.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("ffviz", "Please enter valid numbers for magnitude and years.")
            return
        self.status.set("Loading earthquake list from USGS …")
        threading.Thread(target=self._fetch_events, args=(mag, years), daemon=True).start()

    def _fetch_events(self, mag, years):
        try:
            import requests
            t0 = (datetime.now(timezone.utc) - timedelta(days=365.25 * years)).strftime("%Y-%m-%d")
            r = requests.get(FDSN, params=dict(format="geojson", producttype="finite-fault", starttime=t0,
                                               minmagnitude=mag, orderby="time", limit=500), timeout=60)
            r.raise_for_status()
            feats = r.json().get("features", [])
            rows = []
            for f in feats:
                p = f["properties"]
                t = datetime.fromtimestamp(p["time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                mag_ = p.get("mag")
                rows.append((f["id"], t, f"{mag_:.1f}" if mag_ is not None else "", p.get("place") or ""))
            self.q.put(("events", rows))
        except Exception as e:
            self.q.put(("events_err", str(e)))

    def _on_select(self, _e=None):
        sel = self.tree.selection()
        if sel:
            self.eid.set(sel[0])

    # ------------------------------------------------------------------ run
    def run(self):
        if self.busy:
            return
        eid = self.eid.get().strip()
        if not eid:
            messagebox.showinfo("ffviz", "Please select an earthquake or enter an event ID first.")
            return
        if not _valid_id(eid):
            messagebox.showerror("ffviz", "An event ID contains only letters, digits, '_' and '-' "
                                          "(e.g. us6000u18k).")
            return
        outdir = self.outdir.get().strip() or default_outdir()
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("ffviz", f"Cannot create the output folder:\n{outdir}\n\n{e}")
            return
        out = os.path.join(outdir, f"{eid}_ffm.{self.fmt.get().lower()}")
        argv = [eid, "--out", out, "--days", self.days.get().strip() or "30",
                "--color-by", "depth" if self.color.get() == "depth" else "time"]
        mm = self.minmag.get().strip().replace(",", ".")
        if mm:
            argv += ["--minmag", mm]
        if not self.plates.get():
            argv.append("--no-plates")
        if not self.coast.get():
            argv.append("--no-coast")
        if self.refresh.get():
            argv.append("--refresh")
        self.busy = True
        self.go.configure(state="disabled")
        self.prog.start(12)
        self.log.delete("1.0", "end")
        self.status.set(f"Creating figure for {eid} … (the first run can take 1–2 minutes)")
        threading.Thread(target=self._worker, args=(argv, out), daemon=True).start()

    def _worker(self, argv, out):
        w = _QueueWriter(self.q)
        try:
            with redirect_stdout(w), redirect_stderr(w):
                rc = cli_main(argv)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
        except Exception:
            self.q.put(("log", traceback.format_exc()))
            rc = 99
        self.q.put(("done", (rc, out)))

    # ------------------------------------------------------------------ queue / UI updates
    def _poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "log":
                    self._append_log(data)
                elif kind == "events":
                    self.tree.delete(*self.tree.get_children())
                    for eid, t, m, place in data:
                        self.tree.insert("", "end", iid=eid, values=(t, m, place))
                    self.status.set(f"{len(data)} earthquakes with a finite-fault model. "
                                    "Select one and click “Create figure” (or double-click).")
                elif kind == "events_err":
                    self.status.set("Could not load the list (internet connection?). You can also type an event ID directly.")
                    self._append_log(f"Error while loading the list: {data}\n", "err")
                elif kind == "done":
                    self._finish(*data)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _append_log(self, text, tag=None):
        for line in text.splitlines(keepends=True):
            t = tag
            for key in ("PASS", "WARN", "FAIL"):
                if f"[{key}]" in line:
                    t = key
            if line.lower().startswith("error") or "Traceback" in line:
                t = "err"
            self.log.insert("end", line, t)
        self.log.see("end")

    def _finish(self, rc, out):
        self.busy = False
        self.prog.stop()
        self.go.configure(state="normal")
        if rc == 0 and os.path.exists(out):
            self.last_file = out
            self.b_open.configure(state="normal")
            self.status.set(f"Done: {out}")
            self._img_src = out if out.lower().endswith(".png") else None
            if self._img_src is None:
                self.canvas.delete("all")
                self.canvas.create_text(20, 20, anchor="nw", text="PDF/SVG created – click “Open file” to view it.",
                                        font=("Segoe UI", 11))
            self._show_image()
        else:
            msgs = {3: "USGS has not published a finite-fault model for this event.",
                    2: "Data could not be downloaded (internet connection or wrong event ID?).",
                    4: "The model file could not be read – please report the log below.",
                    5: "A consistency check failed – see the log below.",
                    6: "The output file is open in another program – close it and try again."}
            self.status.set("Error: " + msgs.get(rc, "see the log below."))

    def _show_image(self):
        if not self._img_src:
            return
        try:
            from PIL import Image, ImageTk
            im = Image.open(self._img_src)
            cw, ch = max(50, self.canvas.winfo_width()), max(50, self.canvas.winfo_height())
            scale = min(cw / im.width, ch / im.height)
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
            self._img = ImageTk.PhotoImage(im)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._img, anchor="center")
        except Exception as e:
            self._append_log(f"Preview not available: {e}\n", "err")

    # ------------------------------------------------------------------ helpers
    def _pick_dir(self):
        d = filedialog.askdirectory(initialdir=self.outdir.get() or os.path.expanduser("~"))
        if d:
            self.outdir.set(d)

    def _open_file(self):
        if self.last_file and os.path.exists(self.last_file):
            _open_path(self.last_file)

    def _open_dir(self):
        d = self.outdir.get().strip() or default_outdir()
        os.makedirs(d, exist_ok=True)
        _open_path(d)


def _valid_id(eid: str) -> bool:
    import re
    return re.fullmatch(r"[A-Za-z0-9_\-]{2,40}", eid) is not None


def _open_path(path):
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 (Windows only)
    elif sys.platform == "darwin":
        import subprocess
        subprocess.Popen(["open", path])
    else:
        import subprocess
        subprocess.Popen(["xdg-open", path])


def main():
    if sys.platform == "win32":               # sharp text on high-DPI screens
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    App().mainloop()


if __name__ == "__main__":
    main()
