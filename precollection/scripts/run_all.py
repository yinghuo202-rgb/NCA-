from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    scripts_dir = Path(__file__).resolve().parent
    steps = [
        "00_setup.py",
        "01_build_manifest.py",
        "02_download_subtitles.py",
        "03_parse_subtitles.py",
        "04_clean_text.py",
        "05_compute_features.py",
        "06_generate_report.py",
    ]
    for step in steps:
        script_path = scripts_dir / step
        print(f"\n=== RUN {step} ===", flush=True)
        subprocess.run([sys.executable, str(script_path)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
