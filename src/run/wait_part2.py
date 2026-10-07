"""Start Part 2 after the scene split and the viewer import both exist."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.run.launch_part2 import execute_part2


def wait_for_part2(split_path: Path, viewer_log: Path, interval_seconds: int = 60) -> None:
    while True:
        ready = split_path.is_file() and viewer_log.is_file() and "VIEWER_IMPORT_OK" in viewer_log.read_text()
        if ready:
            return
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--viewer-log", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, default=None)
    args = parser.parse_args()
    wait_for_part2(args.prepared / "split.json", args.viewer_log)
    for split_name in ("dense", "sparse"):
        execute_part2(args.prepared, args.work, split_name, args.python, args.cameras)


if __name__ == "__main__":
    main()
