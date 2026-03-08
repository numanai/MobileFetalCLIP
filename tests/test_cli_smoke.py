from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cli_help_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, "-m", "mobile_fetal_clip.cli", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "MobileFetalCLIP CLI" in proc.stdout
