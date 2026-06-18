from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_retro.scripts.import_path import main as stable_retro_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import ROMs into stable-retro's data directory")
    parser.add_argument("rom_path", nargs="?", default="~/Desktop/roms")
    parser.add_argument(
        "--game",
        help="Optional Stable Retro game id to verify after import.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rom_path = Path(args.rom_path).expanduser()
    sys.argv = ["stable_retro.import", str(rom_path)]
    stable_retro_import()
    if args.game:
        import stable_retro as retro

        imported = retro.data.get_romfile_path(args.game)
        print(f"{args.game} imported at {imported}")
    else:
        print(f"ROM import finished from {rom_path}")


if __name__ == "__main__":
    main()
