#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import w0waCDM


from clens.util import constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters
from clens.util.survey import Survey


"""
calaculating lensing kernel, making various sanity checks 
"""

class LensingKernel(object):
    def __init__(self, co, su):
        self.co = co
        self.su = su

        # astropy
        #astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        astropy_dist = w0waCDM(H0=self.co.h*100, Om0=self.co.OmegaM, Ode0=self.co.OmegaDE,
            w0=self.co.w0, wa=self.co.wa)
        # Astropy 8 makes the redshift argument positional-only.
        self.chi = lambda z: astropy_dist.comoving_distance(z)


        self.calc_kernel()

    def calc_kernel(self):
        zl_min = 0.1
        # if self.su.zs_max is None:
        #     zs_max = 5.
        # else:
        if self.su.top_hat == True:
            print('using tophat source distribution')

        zs_max = self.su.zs_max
        zl_list = np.linspace(zl_min, zs_max-0.01, 100)
        kernel_list = np.zeros(len(zl_list))
        for iz, zl in enumerate(zl_list):
            #if self.su.zs_min is None:
            zs_list = np.linspace(max(zl+0.01,self.su.zs_min), zs_max, 100)
            #else:
            #    zs_list = np.linspace(self.su.zs_min, self.su.zs_max, 100)
                
            ns_list = self.su.pz_src(zs_list)
            chi_l = self.chi(z=zl).value
            chi_s = self.chi(z=zs_list).value
            #print('zl', zl, zs_list)
            Sigma_crit = (cn.c)**2/(4.*np.pi*cn.G)*chi_s / chi_l /(chi_s - chi_l)/(1.+zl) #Msun/Mpc^2
            integrand = ns_list/Sigma_crit
            kernel_list[iz] = np.trapz(integrand, x=zs_list)
        self.kernel_z_interp = interp1d(zl_list, kernel_list)


    def calc_kernel_Sigma(self, zh): # used for C_ell_Sigma # depend on redshift of cluster # ugly!!!
        chi_h = self.chi(z=zh).value
        #print('kernel for Sigma')

        zl_min = 0.1
        if self.su.top_hat == True:
            print('lensing kernel uses top-hat source distribution')

        #zs_max = self.su.zs_max
        zl_list = np.linspace(zl_min, self.su.zs_max-0.01, 100)
        kernel_list = np.zeros(len(zl_list))
        for iz, zl in enumerate(zl_list):
            zs_list = np.linspace(max(zl+0.01,self.su.zs_min), self.su.zs_max, 100)
            ns_list = self.su.pz_src(zs_list)
            chi_l = self.chi(z=zl).value
            chi_s = self.chi(z=zs_list).value
            #print('zl', zl, zs_list)
            Sigma_crit = (cn.c)**2/(4.*np.pi*cn.G)*chi_s / chi_l /(chi_s - chi_l)/(1.+zl) #Msun/Mpc^2
            Sigma_crit_halo = (cn.c)**2/(4.*np.pi*cn.G)*chi_s / chi_h /(chi_s - chi_h)/(1.+zh) #Msun/Mpc^2
            integrand = ns_list / Sigma_crit * Sigma_crit_halo
            kernel_list[iz] = np.trapz(integrand, x=zs_list)
        self.kernel_Sigma_z_interp = interp1d(zl_list, kernel_list)


    def mean_Sigma_crit(self, zh):
        chi_h = self.chi(z=zh).value
        zs_list = np.linspace(max(zh+0.01,self.su.zs_min), self.su.zs_max, 100)
        ns_list = self.su.pz_src(zs_list)
        chi_s = self.chi(z=zs_list).value
        Sigma_crit_halo = (cn.c)**2/(4.*np.pi*cn.G)*chi_s / chi_h /(chi_s - chi_h)/(1.+zh) #Msun/Mpc^2
        integrand = ns_list * Sigma_crit_halo
        return np.trapz(integrand, x=zs_list)

    def fsrc_behind_zh(self, zh):
        zs_list = np.linspace(max(zh+0.01,self.su.zs_min), self.su.zs_max, 100)
        ns_list = self.su.pz_src(zs_list)
        return np.trapz(ns_list, x=zs_list)

if __name__ == "__main__":
    co = CosmoParameters()
    su = Survey()
    lk = LensingKernel(co=co, su=su)
    print("kernel at z=0.3:", lk.kernel_z_interp(0.3))
