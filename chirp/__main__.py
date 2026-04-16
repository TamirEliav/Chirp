"""Entry point for `python -m chirp`.

Delegates to the package-level `main()` function defined in
`chirp/__init__.py`. Kept minimal so both invocation styles — the
legacy `python chirp.py` (via the root shim) and `python -m chirp` —
end up calling the same code path.
"""

from chirp import main

if __name__ == "__main__":
    main()
