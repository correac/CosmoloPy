"""
Fitting Formulae for CDM + Baryon + Massive Neutrino (MDM) cosmologies
from the Eisenstein & Hu fitting formula (from Eisenstein & Hu, 1999 ApJ 511 5),
based on the cosmology currently held in the internal variables.

There are two primary functions in the class, one to set the cosmology
(__init()__), the other to construct the transfer function for a
wavenumber k (tf_k_mpc or tf_k_hmpc). These functions return tf_cb
(the CDM+Baryon density-weighted transfer function) and tf_cbnu (the CDM+
Baryon+Neutrino density-weighted transfer function).

Please note that all internal length scales are in Mpc, not h^-1 Mpc!
"""
import numpy as np


class TransferFunction:
    """ This routine takes cosmological parameters and a redshift and sets up
        all the internal scalar quantities needed to compute the transfer function.
        
    Parameters
    ----------
    cosmo : dict
    
    Specify the cosmological parameters with the keys 'omega_M_0',
    'omega_b_0', 'omega_n_0', 'N_nu', 'omega_lambda_0', 'h' and
    'baryonic_effects'.

    omega_matter -- Density of CDM, baryons, and massive neutrinos,
    (in units of the critical density).
    omega_baryon -- Density of baryons, in units of critical.
    omega_hdm    -- Density of massive neutrinos, in units of critical
    omega_lambda -- Cosmological constant
    hubble       -- Hubble constant, in units of 100 km/s/Mpc
    N_nu         -- (Int) Number of degenerate massive neutrino species
    
    redshift : scalar/array -- The redshift to evaluate
    
    Returns
    -------
        
    Sets many global variables for use in tf_k_mpc() """
    
    def __init__(self, redshift, **cosmo):
        
        # Set this for TF_k_mpc(), the routine crashes if baryons or neutrinos are zero #
        self.num_degen_hdm = int(cosmo['N_nu']) # Number of degenerate massive neutrino species #
        self.omega_baryon = cosmo['omega_b_0']
        self.omega_hdm = cosmo['omega_n_0']
        if self.num_degen_hdm<1: self.num_degen_hdm=1.0
        if self.omega_baryon<=0: self.omega_baryon=1e-5
        if self.omega_hdm<=0: self.omega_hdm=1e-5
        
        self.theta_cmb = 2.728/2.7    # The temperature of the CMB, in units of 2.7 K, assuming T_cmb = 2.728 K #
        self.omega_curv = 1.0-cosmo['omega_M_0']-cosmo['omega_lambda_0']
        self.omhh = cosmo['omega_M_0'] * cosmo['h']**2
        self.obhh = self.omega_baryon * cosmo['h']**2
        self.onhh = self.omega_hdm * cosmo['h']**2
        self.f_baryon = self.omega_baryon/cosmo['omega_M_0'] # Baryon fraction #
        self.f_hdm = self.omega_hdm/cosmo['omega_M_0'] # Massive Neutrino fraction #
        self.f_cdm = 1.0-self.f_baryon-self.f_hdm # CDM fraction #
        self.f_cb = self.f_cdm+self.f_baryon # Baryon + CDM fraction #
        self.f_bnu = self.f_baryon+self.f_hdm # Baryon + Massive Neutrino fraction #
        
        # Compute the equality scale. #
        # z_equality Redshift of matter-radiation equality, eq. (1) #
        self.z_equality = 25000.0*self.omhh/self.theta_cmb**4   # Actually 1+z_eq #
        self.k_equality = 0.0746*self.omhh/self.theta_cmb**2 # The comoving wave number of the horizon at equality #
        
        # Compute the drag epoch and sound horizon, eq. (2) #
        z_drag_b1 = 0.313 * self.omhh**(-0.419) * (1+0.607 * self.omhh**0.674)
        z_drag_b2 = 0.238 * self.omhh**0.223
        # z_drag Redshift of the drag epoch #
        self.z_drag = 1291. * self.omhh**0.251/(1.0+0.659 * self.omhh**0.828)
        self.z_drag *= (1.0 + z_drag_b1 * self.obhh**(z_drag_b2))
        y_drag = self.z_equality / (1.0+self.z_drag)
        
        # The sound horizon at the drag epoch, units Mpc, eq. (4) #
        self.sound_horizon_fit = 44.5 * np.log(9.83/self.omhh) / np.sqrt(1.0+10.0 * self.obhh**(0.75))
        
        # Set up for the free-streaming & infall growth function #
        self.p_c = 0.25 * (5.0-np.sqrt(1.0+24.0 * self.f_cdm)) # The correction to the exponent before drag epoch #
        self.p_cb = 0.25 * (5.0-np.sqrt(1.0+24.0 * self.f_cb)) # The correction to the exponent after drag epoch #
        
        omega_denom = cosmo['omega_lambda_0']
        omega_denom += (1.0+redshift)**2*(self.omega_curv+cosmo['omega_M_0']*(1.0+redshift))
        
        # Omega_lambda at the given redshift #
        self.omega_lambda_z = cosmo['omega_lambda_0']/omega_denom
        # Omega_matter at the given redshift #
        self.omega_matter_z = cosmo['omega_M_0'] * (1.0+redshift)**3/omega_denom
        
        # D_1(z) -- the growth function as k->0 #
        self.growth_k0 = self.z_equality/(1.0+redshift) * 2.5 * self.omega_matter_z
        ratio = self.omega_matter_z**(4.0/7.0)-self.omega_lambda_z
        ratio += (1.0+self.omega_matter_z/2.0)*(1.0+self.omega_lambda_z/70.0)
        self.growth_k0 /= ratio
        
        # D_1(z)/D_1(0) -- the growth relative to z=0  #
        self.growth_to_z0 = self.z_equality * 2.5 * cosmo['omega_M_0']
        ratio = cosmo['omega_M_0']**(4.0/7.0)-cosmo['omega_lambda_0']
        ratio += (1.0+cosmo['omega_M_0']/2.0) * (1.0+cosmo['omega_lambda_0']/70.0)
        self.growth_to_z0 /= ratio
        
        self.growth_to_z0 = self.growth_k0/self.growth_to_z0
        
        # Compute small-scale suppression #
        self.alpha_nu = self.f_cdm / self.f_cb * (5.0-2.*(self.p_c+self.p_cb)) / (5.-4.*self.p_cb)
        self.alpha_nu *= (1.+y_drag)**(self.p_cb-self.p_c)*(1.+self.f_bnu*(-0.553+0.126*self.f_bnu*self.f_bnu))
        self.alpha_nu /= (1.-0.193*np.sqrt(self.f_hdm*self.num_degen_hdm)+0.169*self.f_hdm*self.num_degen_hdm**(0.2))
        self.alpha_nu *= (1.+(self.p_c-self.p_cb)/2.*(1.+1./(3.-4.*self.p_c)/(7.-4.*self.p_cb))/(1.+y_drag))
        self.alpha_gamma = np.sqrt(self.alpha_nu)
        
        # The correction to the log in the small-scale #
        self.beta_c = 1./(1.-0.949*self.f_bnu)
        self.hubble = cosmo['h']    # Need to pass Hubble constant to TF_k_hmpc() #
    
    
    def tf_k_mpc(self,kk):
        """
        Parameters
        ----------
        kk : array. Wavenumber in Mpc^-1.
            
        Returns
        -------
        The following are set as global variables:
        growth_cb   -- the transfer function for density-weighted CDM + Baryon perturbations.
        growth_cbnu -- the transfer function for density-weighted CDM + Baryon + Massive Neutrino perturbations.
        """
        # Wavenumber rescaled by \Gamma #
        qq = kk / self.omhh * self.theta_cmb**2
        
        # The epoch of free-streaming for a given scale #
        y_freestream = 17.2 * self.f_hdm * (1.+0.488 * self.f_hdm**(-7.0/6.0))
        y_freestream *= (self.num_degen_hdm * qq / self.f_hdm)**2
        
        temp1 = self.growth_k0**(1.0-self.p_cb)
        temp2 = (1.0+self.growth_k0)/(1.0+y_freestream)
        temp3 = self.growth_k0/(1.0+y_freestream)
        # Growth factor for CDM+Baryon perturbations #
        growth_cb = temp2**(self.p_cb/0.7) * temp1
        # Growth factor for CDM+Baryon+Neutrino pert. #
        growth_cbnu = (self.f_cb**(0.7/self.p_cb)+temp3)**(self.p_cb/0.7) * temp1
        
        # Compute the master function #
        ratio = 1.+(kk*self.sound_horizon_fit*0.43)**4
        gamma_eff = self.omhh * (self.alpha_gamma+(1.-self.alpha_gamma)/ratio)
        # Wavenumber rescaled by effective Gamma #
        qq_eff = qq * self.omhh / gamma_eff
        
        # Calculate Suppressed TF #
        tf_sup_L = np.log(np.e+1.84 * self.beta_c * self.alpha_gamma * qq_eff)
        tf_sup_C = 14.4 + 325./(1.0+60.5*qq_eff**(1.11))
        tf_sup = tf_sup_L / (tf_sup_L+tf_sup_C*qq_eff**2)
        
        # Wavenumber compared to maximal free streaming #
        qq_nu = 3.92 * qq * np.sqrt(self.num_degen_hdm)/self.f_hdm
        # Correction near maximal free streaming #
        ratio = qq_nu**(-1.6) + qq_nu**(0.8)
        max_fs_correction = 1. + 1.2*(self.f_hdm**(0.64)*self.num_degen_hdm**(0.3+0.6*self.f_hdm))/ratio
        tf_master = tf_sup * max_fs_correction
        
        # Now compute the CDM+HDM+baryon transfer functions #
        tf_cb = tf_master * growth_cb / self.growth_k0
        tf_cbnu = tf_master * growth_cbnu / self.growth_k0
        
        # tf_cb: The transfer function for density-weighted CDM + Baryon perturbations #
        # tf_cbnu: The transfer function for density-weighted CDM + Baryon + Massive Neutrino perturbations. #
        return tf_cb, tf_cbnu

    def tf_k_hmpc(self,kk):
        """
        Parameters
        ----------
        kk : array. Wavenumber in h Mpc^-1.

        Returns
        -------
        The following are set as global variables:
        growth_cb   -- the transfer function for density-weighted CDM + Baryon perturbations.
        growth_cbnu -- the transfer function for density-weighted CDM + Baryon + Massive
        Neutrino perturbations.
        """
        
        return self.tf_k_mpc(kk * self.hubble)
    
