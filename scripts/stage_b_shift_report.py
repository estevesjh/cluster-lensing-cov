#!/usr/bin/env python3
"""Stage-B physics-shift report: frozen (EH98) vs live clenspy (CAMB).

Compares two covariance products of scripts/reproduce_covariance.py and
writes a per-term, per-bin ratio summary.  These shifts are EXPECTED
physics changes (EH98 -> CAMB transfer function, BAO wiggles, broadband
amplitude), not regressions — Stage-A already proved the numerics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TERMS = (
    "covariance",
    "covariance_cosmic_shear",
    "covariance_shape_noise",
    "covariance_cross",
)


def report(frozen_path: Path, clenspy_path: Path, output: Path) -> dict:
    a = np.load(frozen_path / "covariance.npz")
    b = np.load(clenspy_path / "covariance.npz")
    labels = [str(x) for x in a["block_labels"]]
    nrad = a["covariance"].shape[0] // len(labels)

    out: dict = {"frozen": str(frozen_path), "clenspy": str(clenspy_path),
                 "n_radial": nrad, "terms": {}}
    for term in TERMS:
        per_bin = []
        for ib, label in enumerate(labels):
            sl = slice(ib * nrad, (ib + 1) * nrad)
            d_a = np.diag(a[term][sl, sl])
            d_b = np.diag(b[term][sl, sl])
            keep = d_a > 0
            ratio = d_b[keep] / d_a[keep]
            per_bin.append(
                dict(
                    block=label,
                    median_ratio=float(np.median(ratio)),
                    min_ratio=float(ratio.min()),
                    max_ratio=float(ratio.max()),
                )
            )
        med = [pb["median_ratio"] for pb in per_bin]
        out["terms"][term] = dict(
            overall_median=float(np.median(med)),
            per_bin=per_bin,
        )

    if "counts_covariance" in a.files and "counts_covariance" in b.files:
        ca, cb = a["counts_covariance"], b["counts_covariance"]
        out["counts_covariance_diag_ratio"] = (
            np.diag(cb) / np.diag(ca)
        ).tolist()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))

    print(f"Stage-B shift report -> {output}")
    for term in TERMS:
        t = out["terms"][term]
        print(f"  {term:30s} median clenspy/frozen = "
              f"{t['overall_median']:.4f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", default="output/covariance_v2")
    parser.add_argument("--clenspy", default="output/covariance_clenspy")
    parser.add_argument(
        "--output", default="output/stage_b_shift_report.json"
    )
    args = parser.parse_args()
    report(Path(args.frozen), Path(args.clenspy), Path(args.output))


if __name__ == "__main__":
    main()
