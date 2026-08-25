"""Assemble per-bin covariance blocks into the full survey matrices.

Reproduces the legacy ``reproduce_covariance.py`` output schema exactly:
z-major, richness-fast block ordering; per-term matrices
(``covariance_cosmic_shear`` / ``_shape_noise`` / ``_cross`` /
``covariance_comoving``); comoving -> physical a^-4 rescale; same npz
keys.  Adds the N_ij covariance (Poisson + sample variance).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .counts_cov import NCountsCov
from .gaussian_delta_sigma import GaussianDeltaSigmaCov
from .inputs import FrozenTables

__all__ = ["CovarianceAssembler"]


@dataclass
class CovarianceAssembler:
    """Drive the full multi-bin covariance from a contract provider."""

    tables: FrozenTables
    n_radial: int
    radial_range_physical_mpc: tuple[float, float]
    q: float = 1.0
    radial_mode: str = "physical"  # or "comoving": range already comoving Mpc

    def run(self, output_dir: str | Path | None = None) -> dict:
        t = self.tables
        samples = t.samples
        z_pairs = sorted({(s.z_min, s.z_max) for s in samples})
        lam_pairs = sorted({(s.lam_min, s.lam_max) for s in samples})
        nz, nlam, nrad = len(z_pairs), len(lam_pairs), self.n_radial
        nblocks = nz * nlam
        shape = (nblocks * nrad, nblocks * nrad)
        cov_comoving = np.zeros(shape)
        cov_cosmic = np.zeros(shape)
        cov_shape = np.zeros(shape)
        cov_cross = np.zeros(shape)
        radii_phys = np.zeros((nz, nrad))
        radii_comoving = np.zeros((nz, nrad))
        labels = []

        engine = GaussianDeltaSigmaCov(
            t.cosmology, t.source, t.geometry, q=self.q
        )
        rmin_phys, rmax_phys = self.radial_range_physical_mpc

        sample_by_bin = {
            (s.z_min, s.z_max, s.lam_min, s.lam_max): s for s in samples
        }
        for iz, (zmin, zmax) in enumerate(z_pairs):
            zmid = 0.5 * (zmin + zmax)
            a = 1.0 / (1.0 + zmid)
            for ilam, (lmin, lmax) in enumerate(lam_pairs):
                s = sample_by_bin[(zmin, zmax, lmin, lmax)]
                block = iz * nlam + ilam
                sl = slice(block * nrad, (block + 1) * nrad)
                labels.append(f"z{iz}_lambda{ilam}")

                if self.radial_mode == "comoving":
                    rp_lo, rp_hi = rmin_phys, rmax_phys
                else:
                    rp_lo, rp_hi = rmin_phys / a, rmax_phys / a
                blocks = engine.compute(s, rp_min=rp_lo, rp_max=rp_hi, n_rp=nrad)
                cov_cosmic[sl, sl] = blocks.cosmic_shear / a**4
                cov_shape[sl, sl] = blocks.shape_noise / a**4
                cov_cross[sl, sl] = blocks.cross / a**4
                cov_comoving[sl, sl] = blocks.total
                radii_comoving[iz] = blocks.rp_mid
                radii_phys[iz] = blocks.rp_mid * a

        covariance = cov_cosmic + cov_shape + cov_cross
        counts_cov = NCountsCov(
            [
                sample_by_bin[(zp[0], zp[1], lp[0], lp[1])]
                for zp in z_pairs
                for lp in lam_pairs
            ]
        )
        result = dict(
            covariance=covariance,
            covariance_cosmic_shear=cov_cosmic,
            covariance_shape_noise=cov_shape,
            covariance_cross=cov_cross,
            covariance_comoving=cov_comoving,
            radii_phys_mpc=radii_phys,
            radii_comoving_mpc_noh=radii_comoving,
            block_labels=np.asarray(labels),
            counts_covariance=counts_cov.matrix(),
            counts_labels=np.asarray(counts_cov.labels()),
        )
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "covariance.npz", **result)
            np.savetxt(out / "covariance.txt", covariance)
            np.savetxt(out / "covariance_shape_noise.txt", cov_shape)
            np.savetxt(out / "covariance_cosmic_shear.txt", cov_cosmic)
            np.savetxt(out / "covariance_cross.txt", cov_cross)
            np.savetxt(out / "radii_phys_mpc.txt", radii_phys)
            np.savetxt(out / "counts_covariance.txt", result["counts_covariance"])
            (out / "metadata.json").write_text(
                json.dumps(
                    dict(
                        engine="fftlog (clenspy.utils.fftlog_cov)",
                        q=self.q,
                        n_radial=nrad,
                        radial_range_physical_mpc=list(
                            self.radial_range_physical_mpc
                        ),
                        provider=t.meta,
                        ordering="z-major, richness-fast, radial-fast",
                    ),
                    indent=2,
                )
            )
        return result
