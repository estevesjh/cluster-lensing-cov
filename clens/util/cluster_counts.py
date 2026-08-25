#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfc
from scipy.interpolate import interp1d
#from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import w0waCDM

from clens.ying.param_w0wa import CosmoParams
from clens.ying.lineartheory import LinearTheory
from clens.ying.density_w0wa import Density
from clens.ying.halostat import HaloStat

from clens.util import constants as cn
from clens.util.parameters import CosmoParameters
from clens.util.scaling_relation import RichnessSelection, FiducialScalingRelation, Costanzi21ScalingRelation, To21ScalingRelation

class ClusterCounts(object):
    def __init__(self, cosmo_parameters, scaling_relation):
        self.cp = cosmo_parameters
        self.sr = scaling_relation
        #print(nuisance_parameters)

    def calc_counts(self, zmin, zmax, lambda_min, lambda_max, survey_area_sq_deg):
        fsky = survey_area_sq_deg/41253.
        #cosmo = FlatLambdaCDM(H0=self.cp.h*100, Om0=self.cp.OmegaM)
        cosmo = w0waCDM(H0=self.cp.h*100, Om0=self.cp.OmegaM, Ode0=self.cp.OmegaDE,
            w0=self.cp.w0, wa=self.cp.wa)
        vol = fsky * 4. * np.pi/3. * (cosmo.comoving_distance(zmax).value**3 - cosmo.comoving_distance(zmin).value**3)
        print('vol', vol*1e-9)

        z = 0.5*(zmin+zmax)
        # using ying's mass function for now
        self.cosmo = CosmoParams(omega_M_0=self.cp.OmegaM, omega_b_0=self.cp.OmegaB, omega_lambda_0=self.cp.OmegaDE, h=self.cp.h, sigma_8=self.cp.sigma8, n=self.cp.ns, tau=self.cp.tau, w0=self.cp.w0, wa=self.cp.wa)
        DELTA_HALO = 200.0

        # mass function
        Mmin = 1e13
        Mmax = 5e15
        dlnM = 0.01
        lnM_arr = np.arange(np.log(Mmin), np.log(Mmax), dlnM)
        M_arr = np.exp(lnM_arr)
        self.den = Density(cosmo=self.cosmo)
        rho_mean_0 = self.den.rho_mean_z(0.)
        hs = HaloStat(cosmo=self.cosmo, z=z, DELTA_HALO=DELTA_HALO, rho_mean_0=rho_mean_0, mass=M_arr, dlogm=dlnM)
        dndM_arr = hs.mass_function
        bias_arr = hs.bias_function
        dndlnM_arr = dndM_arr * M_arr

        rs = RichnessSelection(scaling_relation=self.sr, lambda_min=lambda_min, lambda_max=lambda_max)

        lnM_selection_arr = rs.lnM_selection(lnM_arr, z)
        self.cluster_number_density = np.trapz(dndlnM_arr*lnM_selection_arr, x=lnM_arr)
        #print('self.cluster_number_density', self.cluster_number_density)
        
        self.counts = self.cluster_number_density * vol
        print('counts', self.counts, f'for {lambda_min}, {lambda_max}, {survey_area_sq_deg}')
        
        # sample variance
        bn = np.trapz(bias_arr*dndlnM_arr*lnM_selection_arr, x=lnM_arr) * vol
        _scale = (3./(4.*np.pi) * vol)**(1./3.)
        f_fgrowth  = self.den.growth_factor
        lin_0 = LinearTheory(cosmo=self.cosmo, z=0, den=self.den, set_warnsig8err=True)
        # spherical mass variance as a function of scale.
        sigma_r_0 = lin_0.sigma_r_0_interp()
        _sigma2_v = (sigma_r_0(_scale) * f_fgrowth(z))**2 
        self.sv = bn**2 * _sigma2_v

        self.cluster_mean_bias = bn/self.counts

        # mean mass
        self.lnM_mean = np.trapz(lnM_arr*dndlnM_arr*lnM_selection_arr, x=lnM_arr) / self.cluster_number_density

        return self.counts, self.sv, self.cluster_mean_bias, self.lnM_mean, self.cluster_number_density 

    def calc_counts_full(self, zmin, zmax, lambda_min, lambda_max,
                         survey_area_sq_deg, sigma_z, n_mass=32,
                         n_redshift=24, ltr_bracket_sigma=6.0):
        """Integrate counts and bias with the full ``S_ij(M,z)`` selection.

        This is the selection-function path used by the covariance driver.
        It keeps the existing Ying HMF/bias implementation, while evaluating
        the outer ``(ln M, z)`` integral with fixed Gauss-Legendre nodes and
        asking the RichnessSelection adapter for

        ``S_ij(M,z) = S_i(M,z) K_j(z)``.

        ``sigma_z`` is the per-richness-bin photo-z scatter used by the
        existing RichnessSelection ``K_j`` kernel.  The returned sample
        variance entry is retained for compatibility with ``calc_counts``;
        the Wu et al. covariance uses the returned count and mean bias.
        """
        if sigma_z <= 0.0:
            raise ValueError("sigma_z must be positive")
        if n_mass < 8 or n_redshift < 8:
            raise ValueError("outer quadrature orders must be at least 8")
        if not hasattr(self.sr, "joint_selection_probability"):
            raise TypeError(
                "calc_counts_full requires a RichnessSelection relation "
                "with joint_selection_probability"
            )

        # Ying's HMF uses the same physical Mpc and solar-mass convention as
        # the original midpoint calculation below.
        self.cosmo = CosmoParams(
            omega_M_0=self.cp.OmegaM,
            omega_b_0=self.cp.OmegaB,
            omega_lambda_0=self.cp.OmegaDE,
            h=self.cp.h,
            sigma_8=self.cp.sigma8,
            n=self.cp.ns,
            tau=self.cp.tau,
            w0=self.cp.w0,
            wa=self.cp.wa,
        )
        cosmo = w0waCDM(
            H0=self.cp.h * 100,
            Om0=self.cp.OmegaM,
            Ode0=self.cp.OmegaDE,
            w0=self.cp.w0,
            wa=self.cp.wa,
        )
        area_sr = float(survey_area_sq_deg) * (np.pi / 180.0) ** 2

        # Pad the true-z integration range so the Gaussian-CDF redshift
        # kernel is effectively zero at both integration boundaries.
        z_lo = max(1.0e-3, float(zmin) - ltr_bracket_sigma * sigma_z)
        z_hi = float(zmax) + ltr_bracket_sigma * sigma_z
        z_nodes_unit, z_weights_unit = np.polynomial.legendre.leggauss(
            int(n_redshift)
        )
        z_nodes = 0.5 * (z_hi - z_lo) * z_nodes_unit + 0.5 * (z_hi + z_lo)
        z_weights = 0.5 * (z_hi - z_lo) * z_weights_unit

        lnM_lo = np.log(1.0e13)
        lnM_hi = np.log(5.0e15)
        lnM_unit, lnM_weights_unit = np.polynomial.legendre.leggauss(
            int(n_mass)
        )
        lnM_arr = (
            0.5 * (lnM_hi - lnM_lo) * lnM_unit
            + 0.5 * (lnM_hi + lnM_lo)
        )
        lnM_weights = 0.5 * (lnM_hi - lnM_lo) * lnM_weights_unit
        M_arr = np.exp(lnM_arr)

        self.den = Density(cosmo=self.cosmo)
        rho_mean_0 = self.den.rho_mean_z(0.0)
        total_count = 0.0
        total_bias_weight = 0.0
        total_mass_weight = 0.0

        for z, wz in zip(z_nodes, z_weights):
            z = float(z)
            hs = HaloStat(
                cosmo=self.cosmo,
                z=z,
                DELTA_HALO=200.0,
                rho_mean_0=rho_mean_0,
                mass=M_arr,
                dlogm=float(lnM_arr[1] - lnM_arr[0]),
            )
            dndlnM = hs.mass_function * M_arr
            bias_arr = hs.bias_function
            selection = np.asarray(
                self.sr.joint_selection_probability(
                    M_arr, z, lambda_min, lambda_max,
                    zmin, zmax, sigma_z,
                ),
                dtype=float,
            )
            if selection.shape != M_arr.shape:
                raise ValueError(
                    "joint_selection_probability must return one value per mass"
                )
            dV_dz_dOmega = cosmo.differential_comoving_volume(z).value
            prefactor = wz * area_sr * dV_dz_dOmega
            mass_integral = np.sum(lnM_weights * dndlnM * selection)
            bias_integral = np.sum(
                lnM_weights * dndlnM * bias_arr * selection
            )
            mean_mass_integral = np.sum(
                lnM_weights * dndlnM * np.log(M_arr) * selection
            )
            total_count += prefactor * mass_integral
            total_bias_weight += prefactor * bias_integral
            total_mass_weight += prefactor * mean_mass_integral

        self.counts = float(total_count)
        self.cluster_mean_bias = float(total_bias_weight / total_count)
        self.lnM_mean = float(total_mass_weight / total_count)
        chi_min = cosmo.comoving_distance(zmin).value
        chi_max = cosmo.comoving_distance(zmax).value
        shell_volume = (chi_max**3 - chi_min**3) / 3.0
        self.cluster_number_density = float(total_count / (area_sr * shell_volume))
        # The legacy tuple's second field is not used by CovDeltaSigma.  A
        # full sample-variance calculation would require the covariance
        # volume/window convention, so leave it as a compatibility marker.
        self.sv = np.nan
        return (
            self.counts,
            self.sv,
            self.cluster_mean_bias,
            self.lnM_mean,
            self.cluster_number_density,
        )

if __name__ == "__main__":
    #cosmo_parameters = CosmoParameters()
    cosmo_parameters = CosmoParameters(h=0.7, OmegaDE=0.724, OmegaM=0.276, sigma8=0.802, w0=-0.8, wa=0.2)
    #scaling_relation = FiducialScalingRelation()
    #scaling_relation = Costanzi21ScalingRelation()
    scaling_relation = To21ScalingRelation()
    cmm = ClusterCounts(cosmo_parameters=cosmo_parameters, scaling_relation=scaling_relation)
    cc = cmm.calc_counts(zmin=0.2, zmax=0.35, lambda_min=20, lambda_max=30, survey_area_sq_deg=1437)
    print(cc)
