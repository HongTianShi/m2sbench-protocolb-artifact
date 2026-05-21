from __future__ import annotations

import argparse

from .research.pipeline import reproduce_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the research-hardened end-to-end reproduction pipeline.")
    ap.add_argument("--paper", default="main")
    args = ap.parse_args()
    reproduce_all(args.paper)


if __name__ == "__main__":
    main()
