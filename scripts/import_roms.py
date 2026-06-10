from __future__ import annotations

import argparse
import sys
from pathlib import Path

import stable_retro as retro
from stable_retro.scripts.import_path import main as stable_retro_import

GAME = "SuperMarioBros-Nes-v0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import ROMs into stable-retro's data directory")
    parser.add_argument("rom_path", nargs="?", default="~/Desktop/roms")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rom_path = Path(args.rom_path).expanduser()
    sys.argv = ["stable_retro.import", str(rom_path)]
    stable_retro_import()
    imported = retro.data.get_romfile_path(GAME)
    print(f"{GAME} imported at {imported}")


if __name__ == "__main__":
    main()
