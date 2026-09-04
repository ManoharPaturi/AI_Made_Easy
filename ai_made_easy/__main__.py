"""Entry point: ``python -m ai_made_easy`` launches the GUI app."""
from __future__ import annotations

import sys


def main() -> int:
    from ai_made_easy.ui.app import run

    return run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
