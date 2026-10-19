"""Run the ContextStream backend."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import uvicorn

    from contextstream.api.app import create_app

    uvicorn.run(create_app(args.config), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
