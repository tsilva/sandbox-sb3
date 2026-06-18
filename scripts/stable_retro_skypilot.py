#!/usr/bin/env python
from __future__ import annotations

import sys

from stable_retro_ppo.skypilot_cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
