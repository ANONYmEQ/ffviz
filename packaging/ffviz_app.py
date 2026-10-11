"""Entry point of the stand-alone program (PyInstaller).

Without arguments the desktop app opens; with arguments it behaves like the
command-line tool, e.g.  ffviz.exe us6000u18k --out panama.png
"""
import sys


def run():
    if len(sys.argv) > 1:
        from ffviz.__main__ import main
        sys.exit(main())
    from ffviz.gui import main
    main()


if __name__ == "__main__":
    run()
