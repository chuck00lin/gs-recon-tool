"""Allow ``python -m gs_recon`` as an alias for the ``gs-recon`` console script."""

from gs_recon.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
