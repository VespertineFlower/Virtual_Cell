from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vc_stage01.data import audit_inputs, save_stage0_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the GOAI Stage 0 data contract.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "stage0")
    parser.add_argument("--missing-threshold", type=float, default=0.697)
    args = parser.parse_args()

    objects, report = audit_inputs(args.data_dir, args.missing_threshold)
    save_stage0_cache(args.output_dir, objects, report)
    print(json.dumps(report["audit"], ensure_ascii=False, indent=2))
    print(f"Stage 0 cache written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
