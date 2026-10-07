"""Wait for a download sentinel, then ingest that scene. Do not poll the trainer."""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

from src.data.ingest import ingest_archive
from src.data.splits import SCENES

SENTINELS = {
    "waymo-405841": ("outputs/download_waymo.log", "WAYMO_OK", "data/external/waymo/405841.zip"),
    "dl3dv-2": ("outputs/download_dl3dv.log", "DL3DV_OK", "data/external/dl3dv/2.zip"),
}


def archive_ready(log_path: Path, token: str, archive: Path) -> bool:
    """True only when the log token belongs to a zip that can be extracted.

    A failed download can still print the token. The file itself has to be a zip.
    """
    if not log_path.is_file() or token not in log_path.read_text():
        return False
    return archive.is_file() and zipfile.is_zipfile(archive)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(SENTINELS))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.scene not in SCENES:
        raise SystemExit(args.scene)
    log_name, token, archive_name = SENTINELS[args.scene]
    root = args.project_root
    archive = root / archive_name
    while not archive_ready(root / log_name, token, archive):
        time.sleep(60)
    ingest_archive(
        args.scene,
        archive,
        root / "data" / "external" / args.scene,
        root / "data" / "prepared" / args.scene,
    )
    print(f"PREPARED {args.scene}")


if __name__ == "__main__":
    main()
