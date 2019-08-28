"""
    Fitting Formulae for CDM + Baryon + Massive Neutrino (MDM) cosmologies.
    Daniel J. Eisenstein & Wayne Hu, Institute for Advanced Study.
    
    There are two primary routines here, one to set the cosmology, the
    other to construct the transfer function for a single wavenumber k.
    You should call the former once (per cosmology) and the latter as
    many times as you want.
    
    TFmdm_set_cosm() -- User passes all the cosmological parameters as
    arguments; the routine sets up all of the scalar quantites needed
    computation of the fitting formula.  The input parameters are:
    1) omega_matter -- Density of CDM, baryons, and massive neutrinos,
    in units of the critical density.
    2) omega_baryon -- Density of baryons, in units of critical.
    3) omega_hdm    -- Density of massive neutrinos, in units of critical
    4) degen_hdm    -- (Int) Number of degenerate massive neutrino species
    5) omega_lambda -- Cosmological constant
    6) hubble       -- Hubble constant, in units of 100 km/s/Mpc
    7) redshift     -- The redshift at which to evaluate.
    
    TFmdm_onek_mpc() -- User passes a single wavenumber, in units of Mpc^-1.
    Routine returns the transfer function from the Eisenstein & Hu
    fitting formula, based on the cosmology currently held in the
    internal variables.  The routine returns T_cb (the CDM+Baryon
    density-weighted transfer function), although T_cbn (the CDM+
    Baryon+Neutrino density-weighted transfer function) is stored
    in the global variable tf_cbnu.
    
    We also supply TFmdm_onek_hmpc(), which is identical to the previous
    routine, but takes the wavenumber in units of h Mpc^-1.
    
    We hold the internal scalar quantities in global variables, so that
    the user may access them in an external program, via "extern" declarations.
    
    Please note that all internal length scales are in Mpc, not h^-1 Mpc!
    
"""
import numpy as np

class Transfer_function(**cosmo,redshift):

/* ------------------------- TFmdm_set_cosm() ------------------------ */
int TFmdm_set_cosm(float omega_matter, float omega_baryon, float omega_hdm,
                   int degen_hdm, float omega_lambda, float hubble, float redshift)

""" This routine takes cosmological parameters and a redshift and sets up
    all the internal scalar quantities needed to compute the transfer function.
    INPUT:
    **cosmo
      omega_matter -- Density of CDM, baryons, and massive neutrinos,
                      (in units of the critical density).
      omega_baryon -- Density of baryons, in units of critical.
      omega_hdm    -- Density of massive neutrinos, in units of critical
      omega_lambda -- Cosmological constant
      hubble       -- Hubble constant, in units of 100 km/s/Mpc
      N_nu         -- (Int) Number of degenerate massive neutrino species
    redshift       -- The redshift to evaluate
    OUTPUT:
    Sets many global variables for use in TFmdm_onek_mpc() """

    def __init__(self, **cosmo, redshift):
        
        theta_cmb = 2.728/2.7;    # Assuming T_cmb = 2.728 K #
        num_degen_hdm = int(cosmo['N_nu'])
        omega_curv = 1.0-cosmo['omega_M_0']-cosmo['omega_lambda_0']
        omhh = cosmo['omega_M_0'] * cosmo['h']**0.5
        obhh = cosmo['omega_b_0'] * cosmo['h']**0.5
        onhh = cosmo['omega_n_0'] * cosmo['h']**0.5
        f_baryon = cosmo['omega_b_0']/cosmo['omega_M_0']
        f_hdm = cosmo['omega_n_0']/cosmo['omega_M_0']
        f_cdm = 1.0-f_baryon-f_hdm
        f_cb = f_cdm+f_baryon
        f_bnu = f_baryon+f_hdm
            
        # Compute the equality scale. #
        z_equality = 25000.0*omhh/SQR(SQR(theta_cmb))    # Actually 1+z_eq #
        k_equality = 0.0746*omhh/SQR(theta_cmb)
            
        # Compute the drag epoch and sound horizon #
        z_drag_b1 = 0.313 * omhh**(-0.419) * (1+0.607 * omhh**(0.674))
        z_drag_b2 = 0.238 * omhh**(0.223)
        z_drag = 1291. * omhh**(0.251)/(1.0+0.659 * omhh**(0.828))
        z_drag *= (1.0 + z_drag_b1 * obhh**(z_drag_b2))
        y_drag = z_equality / (1.0+z_drag)

        sound_horizon_fit = 44.5 * np.log(9.83/omhh) / np.sqrt(1.0+10.0 * obhh**(0.75))

        # Set up for the free-streaming & infall growth function #
        p_c = 0.25 * (5.0-np.sqrt(1+24.0 * f_cdm))
        p_cb = 0.25 * (5.0-np.sqrt(1+24.0*f_cb))
    
        omega_denom = omega_lambda
        omega_denom += sqrt(1.0+redshift)*(omega_curv+cosmo['omega_M_0']*(1.0+redshift))
        
        omega_lambda_z = omega_lambda/omega_denom
        omega_matter_z = cosmo['omega_M_0']*SQR(1.0+redshift)*(1.0+redshift)/omega_denom
            
        growth_k0 = z_equality/(1.0+redshift) * 2.5 * omega_matter_z
        ratio = omega_matter_z**(4.0/7.0)-omega_lambda_z
        ratio += (1.0+omega_matter_z/2.0)*(1.0+omega_lambda_z/70.0)
        growth_k0 /= ratio
            
        growth_to_z0 = z_equality*2.5*cosmo['omega_M_0']
        ratio = cosmo['omega_M_0']**(4.0/7.0)-omega_lambda
        ratio += (1.0+cosmo['omega_M_0']/2.0) * (1.0+omega_lambda/70.0)
        growth_to_z0 /= ratio

        growth_to_z0 = growth_k0/growth_to_z0
                                              
        # Compute small-scale suppression #
        alpha_nu = f_cdm / f_cb * (5.0-2.*(p_c+p_cb)) / (5.-4.*p_cb)
        alpha_nu *= (1.+y_drag)**(p_cb-p_c)*(1.+f_bnu*(-0.553+0.126*f_bnu*f_bnu))
        alpha_nu /= (1.-0.193*np.sqrt(f_hdm*num_degen_hdm)+0.169*f_hdm*num_degen_hdm**(0.2))
        alpha_nu *= (1.+(p_c-p_cb)/2.*(1.+1./(3.-4.*p_c)/(7.-4.*p_cb))/(1+y_drag))
        alpha_gamma = np.sqrt(alpha_nu)
        beta_c = 1/(1-0.949*f_bnu)
        # Done setting scalar variables #
        hhubble = hubble    # Need to pass Hubble constant to TFmdm_onek_hmpc() #


    def TF_k_mpc(self,kk,**cosmo):
    """
       Given a wavenumber in Mpc^-1, return the transfer function for the
       cosmology held in the global variables.
       Input: kk -- Wavenumber in Mpc^-1
       Output: The following are set as global variables:
            growth_cb   -- the transfer function for density-weighted CDM
                           + Baryon perturbations.
            growth_cbnu -- the transfer function for density-weighted
                           CDM + Baryon + Massive Neutrino perturbations.
       The function returns growth_cb
       """

        qq = kk / omhh * np.sqrt(theta_cmb)
    
        # Compute the scale-dependent growth functions #
        y_freestream = 17.2 * f_hdm * (1+0.488 * f_hdm**(-7.0/6.0))
        y_freestream *= np.sqrt(num_degen_hdm * qq / f_hdm)
        temp1 = growth_k0**(1.0-p_cb)
        temp2 = (growth_k0/(1+y_freestream))**(0.7)
        growth_cb = (1.0+temp2)**(p_cb/0.7) * temp1
        growth_cbnu = (f_cb**(0.7/p_cb)+temp2)**(p_cb/0.7) * temp1

        # Compute the master function #
        gamma_eff = omhh * (alpha_gamma+(1.-alpha_gamma)/(1.+np.sqrt(np.sqrt(kk*sound_horizon_fit*0.43))))
        qq_eff = qq * omhh / gamma_eff
                     
        tf_sup_L = np.log(2.71828+1.84 * beta_c * alpha_gamma * qq_eff)
        tf_sup_C = 14.4+325/(1+60.5*qq_eff**(1.11))
        tf_sup = tf_sup_L/(tf_sup_L+tf_sup_C*np.sqrt(qq_eff))
                     
        qq_nu = 3.92 * qq * np.sqrt(num_degen_hdm/f_hdm)
        max_fs_correction = 1.+1.2*(f_hdm**(0.64)*num_degen_hdm**(0.3+0.6*f_hdm)/(qq_nu**(-1.6)+qq_nu**(0.8))
        tf_master = tf_sup * max_fs_correction

        # Now compute the CDM+HDM+baryon transfer functions #
        tf_cb = tf_master * growth_cb / growth_k0
        tf_cbnu = tf_master * growth_cbnu / growth_k0
        return tf_cb
                                    
    def TF_k_hmpc(self,kk,**cosmo):
    """
     Given a wavenumber in h Mpc^-1, return the transfer function (TF) for the
    cosmology held in the global variables.
    Input: kk -- Wavenumber in h Mpc^-1 */
    Output: The following are set as global variables:
        growth_cb   -- the transfer function for density-weighted
                       CDM + Baryon perturbations.
        growth_cbnu -- the transfer function for density-weighted
                       CDM + Baryon + Massive Neutrino perturbations.
    The function returns growth_cb """
    
    return TF_k_mpc(self,kk * cosmo['h']);
    


