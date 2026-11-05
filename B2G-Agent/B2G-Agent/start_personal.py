"""Start the private UI from this checkout, including after a folder rename."""
from pathlib import Path
import os
import sys


def main() -> int:
    root = Path(__file__).resolve().parent
    # Existing Windows console launchers and editable installs may point to an
    # old checkout location. Resolve both source and relative assets here.
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)
    from b2g_agent.web.cli import main as web_main

    return web_main()


if __name__ == "__main__":
    raise SystemExit(main())
