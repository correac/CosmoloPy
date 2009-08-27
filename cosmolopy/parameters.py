"""Provides some pre-defined sets of cosmological parameters."""

def add_extras(cosmo):
    """Sets neutrino number N_nu = 0, neutrino density
       omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.
    """
    extras = {'omega_n_0' : 0.0,
              'N_nu': 0,
             'Y_He': 0.24}

    cosmo.update(extras)
    return cosmo

def WMAP5_BAO_SN_mean(flat=False, extras=True):
    """WMAP5 + BAO + SN parameters from Komatsu et al. (2009ApJS..180..330K).

    Parameters:
    ----------
    
    flat: boolean
    
      If True, sets omega_lambda_0 = 1 - omega_M_0 to ensure omega_k_0
      = 0 exactly. Also sets omega_k_0 = 0 explicitly.

    extras: boolean

      If True, sets neutrino numbe N_nu = 0, neutrino density
      omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.

    Notes:
    -----

    From the abstract of the paper:

      The six parameters and the corresponding 68% uncertainties,
      derived from the WMAP data combined with the distance
      measurements from the Type Ia supernovae (SN) and the Baryon
      Acoustic Oscillations (BAO) in the distribution of galaxies,
      are: 

      Omega_B h^2 = 0.02267+0.00058-0.00059, 
      Omega_c h^2 = 0.1131 +/- 0.0034, 
      Omega_Lambda = 0.726 +/- 0.015, 
      n_s = 0.960 +/- 0.013, 
      tau = 0.084 +/- 0.016, and 
      Delata^2 R = (2.445 +/- 0.096) * 10^-9 at k = 0.002 Mpc^-1. 

      From these, we derive 

      sigma_8 = 0.812 +/- 0.026, 
      H0 = 70.5 +/- 1.3 km s^-11 Mpc^-1, 
      Omega_b = 0.0456 +/- 0.0015, 
      Omega_c = 0.228 +/- 0.013, 
      Omega_m h^2 = 0.1358 + 0.0037 - 0.0036, 
      zreion = 10.9 +/- 1.4, and 
      t0 = 13.72 +/- 0.12 Gyr.

      """
    omega_c_0 = 0.228
    omega_b_0 = 0.0456
    cosmo = {'omega_b_0' : omega_b_0,
             'omega_M_0' : omega_b_0 + omega_c_0,
             'omega_lambda_0' : 0.726,
             'h' : 0.706,
             'n' : 0.960,
             'sigma_8' : 0.812,
             'tau' : 0.084,
             'z_reion' : 10.9,
             't_0' : 13.72
             }
    if flat:
        cosmo['omega_lambda_0'] = 1. - cosmo['omega_M_0']
        cosmo['omega_k_0'] = 0.0
    if extras:
        add_extras(cosmo)
    return cosmo

def WMAP5_ML(flat=False, extras=True):
    """WMAP5 parameters (using WMAP data alone) from Komatsu et
    al. (2009ApJS..180..330K).

    Parameters:
    ----------
    
    flat: boolean
    
      If True, sets omega_lambda_0 = 1 - omega_M_0 to ensure omega_k_0
      = 0 exactly. Also sets omega_k_0 = 0 explicitly.

    extras: boolean

      If True, sets neutrino numbe N_nu = 0, neutrino density
      omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.

    Notes:
    -----

    Values taken from "WMAP 5 Year ML" column of Table 1 of the paper.

      """
    omega_c_0 = 0.206
    omega_b_0 = 0.0432
    cosmo = {'omega_b_0' : omega_b_0,
             'omega_M_0' : omega_b_0 + omega_c_0,
             'omega_lambda_0' : 0.751,
             'h' : 0.724,
             'n' : 0.961,
             'sigma_8' : 0.787,
             'tau' : 0.089,
             'z_reion' : 11.2,
             't_0' : 13.69
             }
    if flat:
        cosmo['omega_lambda_0'] = 1. - cosmo['omega_M_0']
        cosmo['omega_k_0'] = 0.0
    if extras:
        add_extras(cosmo)
    return cosmo

def WMAP5_mean(flat=False, extras=True):
    """WMAP5 parameters (using WMAP data alone) from Komatsu et
    al. (2009ApJS..180..330K).

    Parameters:
    ----------
    
    flat: boolean
    
      If True, sets omega_lambda_0 = 1 - omega_M_0 to ensure omega_k_0
      = 0 exactly. Also sets omega_k_0 = 0 explicitly.

    extras: boolean

      If True, sets neutrino numbe N_nu = 0, neutrino density
      omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.

    Notes:
    -----

    Values taken from "WMAP 5 Year Mean" of Table 1 of the paper.

    """
    omega_c_0 = 0.214
    omega_b_0 = 0.0441
    cosmo = {'omega_b_0' : omega_b_0,
             'omega_M_0' : omega_b_0 + omega_c_0,
             'omega_lambda_0' : 0.742,
             'h' : 0.719,
             'n' : 0.963,
             'sigma_8' : 0.796,
             'tau' : 0.087,
             'z_reion' : 11.0,
             't_0' : 13.69
             }
    if flat:
        cosmo['omega_lambda_0'] = 1. - cosmo['omega_M_0']
        cosmo['omega_k_0'] = 0.0
    if extras:
        add_extras(cosmo)
    return cosmo