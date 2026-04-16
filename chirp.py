"""Legacy entry point for `python chirp.py`.

The real code now lives in the `chirp/` package. This shim exists so the
documented launch command in CLAUDE.md (`python chirp.py`) keeps working
alongside the new `python -m chirp` form. Python resolves the `chirp`
package (a directory with `__init__.py`) in preference to this same-named
module when imported, so `from chirp import main` below pulls from the
package — not recursively from this file.
"""

from chirp import main

if __name__ == "__main__":
    main()
