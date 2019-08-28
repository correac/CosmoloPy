""" The following routines implement all of the fitting formulae in
Eisenstein \& Hu (1997)

There are two sets of routines here.  The first set,
    
    TFfit_hmpc(), TFset_parameters(), and TFfit_onek(),

calculate the transfer function for an arbitrary CDM+baryon universe using
the fitting formula in Section 3 of the paper.  The second set,
    
    TFsound_horizon_fit(), TFk_peak(), TFnowiggles(), and TFzerobaryon(),

calculate other quantities given in Section 4 of the paper.

The following is an example of a driver routine you might use.
Basically, the driver routine needs to call TFset_parameters() to
set all the scalar parameters, and then call TFfit_onek() for each
wavenumber k you desire. */

While the routines use Mpc^-1 units internally, this driver has been
written to take an array of wavenumbers in units of h Mpc^-1.  On the
other hand, if you want to use Mpc^-1 externally, you can do this by
altering the variables you pass to the driver:
    omega0 -> omega0*hubble*hubble, hubble -> 1.0

INPUT:
        omega0 -- the matter density (baryons+CDM) in units of critical
        f_baryon -- the ratio of baryon density to matter density
        hubble -- the Hubble constant, in units of 100 km/s/Mpc
        Tcmb -- the CMB temperature in Kelvin. T<=0 uses the COBE value 2.728.
        numk -- the length of the following zero-offset array
        k[] -- the array of wavevectors k[0..numk-1]  */

INPUT/OUTPUT: There are three output arrays of transfer functions.
All are zero-offset and, if used, must have storage [0..numk-1] declared
in the calling program.  However, if you substitute the NULL pointer for
one or more of the arrays, then that particular transfer function won't
be outputted. The transfer functions are:
    
    tf_full[] -- The full fitting formula, eq. (16), for the matter
        transfer function.
    tf_baryon[] -- The baryonic piece of the full fitting formula, eq. 21.
    tf_cdm[] -- The CDM piece of the full fitting formula, eq. 17. """

import numpy as np

def TFfit_hmpc(omega0, f_baryon, hubble, Tcmb, k):
    # Remember: k is in units of h Mpc^-1. */

    TF = TFfit(omega0*hubble*hubble, f_baryon, Tcmb)
    #tf_full, tf_thisk, tf_baryon, tf_cdm = TF.TFfit_onek(k * hubble)
    return TF.TFfit_onek(k * hubble)

class TFfit:

    def __init__(self, omega0, hubble, f_baryon, Tcmb):
        """
        Parameters
        ----------
        :param omega0: The density of CDM and baryons, in units of critical dens
        :param hubble: Hubble constant, in units of 100 km/s/Mpc
        :param f_baryon: Fraction of baryons to CDM
        :param Tcmb: The temperature of the CMB in Kelvin. Tcmb<=0 forces use
        of the COBE value of  2.728 K.
        """
        # Note: Units are always Mpc, never h^-1 Mpc. #

        self.omhh = omega0 * hubble**2
        self.obhh = self.omhh * f_baryon
        if Tcmb<=0.0: Tcmb=2.728
        theta_cmb = Tcmb/2.7

        z_equality = 2.50e4 * omhh / theta_cmb**4  # Really 1+z #
        self.k_equality = 0.0746 * omhh / theta_cmb**2

        z_drag_b1 = 0.313 * omhh**(-0.419) * (1.0+0.607 * omhh**0.674)
        z_drag_b2 = 0.238 * omhh**0.223
        z_drag = 1291. * omhh**0.251 / (1.0+0.659 * omhh**0.828) * (1.0+z_drag_b1 * obhh**z_drag_b2)

        R_drag = 31.5 * obhh / theta_cmb**4 * (1000./(1.0+z_drag))
        R_equality = 31.5 * obhh / theta_cmb**4 * (1000./z_equality)

        sound_horizon = 2. /3. / k_equality * np.sqrt(6./R_equality)
        sound_horizon *= np.log((np.sqrt(1.0+R_drag)+np.sqrt(R_drag+R_equality))/(1.0+sqrt(R_equality)))

        k_silk = 1.6 * obhh**0.52 * omhh**0.73 * (1.0+(10.4*omhh)**(-0.95))

        alpha_c_a1 = (46.9*omhh)**0.670 * (1.0+(32.1*omhh)**(-0.532))
        alpha_c_a2 = (12.0*omhh)**0.424 * (1.0+(45.0*omhh)**(-0.582))
        alpha_c = alpha_c_a1**(-f_baryon) * alpha_c_a2**(-f_baryon**3)

        beta_c_b1 = 0.944 / (1.0 + (458.*omhh)**(-0.708))
        beta_c_b2 =  (0.395 * omhh)**(-0.0266)
        beta_c = 1.0 / (1.0+beta_c_b1 * ((1.0-f_baryon)**beta_c_b2)-1.0)

        y = z_equality / (1.0 + z_drag)
        alpha_b_G = y * (-6.*np.sqrt(1.0+y)+(2.+3.*y)*np.log((np.sqrt(1.0+y)+1.0)/(np.sqrt(1.0+y)-1.0)))
        alpha_b = 2.07 * k_equality * sound_horizon * (1.0+R_drag)**(-0.75) * alpha_b_G

        beta_node = 8.41 * omhh**0.435
        beta_b = 0.5+f_baryon+(3.-2.*f_baryon) * np.sqrt((17.2*omhh)**2+1.0)

        k_peak = 2.5 * np.pi * (1.0+0.217 * omhh) / sound_horizon #sound_horizon_fit I think!#
        sound_horizon_fit = 44.5 * np.log(9.83/omhh)/np.sqrt(1.0+10.0 * obhh**0.75)

        alpha_gamma = 1.0-0.328 * np.log(431.0*omhh) * f_baryon + 0.38 * np.log(22.3*omhh) * (f_baryon)**2
    
    def tffit_k(self,k):
        """
        Input: k -- Wavenumber at which to calculate transfer function, in Mpc^-1.
        Output: Returns the value of the full transfer function fitting formula.
        This is the form given in Section 3 of Eisenstein & Hu (1997)."""
        # Notes: Units are Mpc, not h^-1 Mpc. #

        k = np.abs(k)    # Just define negative k as positive #

        q = k / 13.41 / k_equality
        xx = k * sound_horizon

        T_c_ln_beta = np.log(np.e+1.8*beta_c*q)
        T_c_ln_nobeta = np.log(np.e+1.8*q)
        T_c_C_alpha = 14.2 / alpha_c + 386.0 / (1.0+69.9 * q**1.08)
        T_c_C_noalpha = 14.2 + 386.0/(1.0+69.9 * q**1.08)

        T_c_tilde = T_c_ln_beta / (T_c_ln_beta+T_c_C_noalpha * q**2)
        T_c_f = 1.0/(1.0+(xx/5.4)**4))
        T_c = T_c_f * T_c_tilde + (1.0-T_c_f) * T_c_tilde

        s_tilde = sound_horizon / (1.0+(beta_node/xx)**3)**(1./3.)
        xx_tilde = k * s_tilde

        T_b_T0 = T_c_tilde
        T_b = np.sin(xx_tilde) / (xx_tilde)
        T_b *= (T_b_T0/(1.0+(xx/5.2)**2) + alpha_b / (1.0+(beta_b/xx)**3) * np.exp(-1.0 * (k/k_silk)**1.4))

        f_baryon = obhh / omhh
        T_full = f_baryon*T_b + (1.0-f_baryon) * T_c

        return T_full, T_b, T_c #Full, baryon, CDM#

    # ======================= Approximate forms =========================== #

    def tf_sound_horizon_fit(self, omega0, f_baryon, hubble):
    """ Input:
        omega0 -- CDM density, in units of critical density
        f_baryon -- Baryon fraction, the ratio of baryon to CDM density.
        hubble -- Hubble constant, in units of 100 km/s/Mpc
        Output: The approximate value of the sound horizon, in h^-1 Mpc. */
        !Note: If you prefer to have the answer in  units of Mpc, use hubble -> 1
        and omega0 -> omega0*hubble^2."""

        omhh = omega0 * hubble * hubble
        obhh = omhh * f_baryon
        sound_horizon_fit = 44.5 * np.log(9.83/omhh)/np.sqrt(1.0+10.0 * obhh**0.75)
        return sound_horizon_fit * hubble

    def tfk_peak(self, omega0, f_baryon, hubble):
        """
        :param self:
        :param omega0: CDM density, in units of critical density
        :param f_baryon: Baryon fraction, the ratio of baryon to CDM density.
        :param hubble: Hubble constant, in units of 100 km/s/Mpc
        :return: The approximate location of the first baryonic peak, in h Mpc^-1
        """
        omhh = omega0 * hubble * hubble
        k_peak_mpc = 2.5 * np.pi * (1.0+0.217*omhh) / tf_sound_horizon_fit(omhh,f_baryon,1.0)
        return k_peak_mpc / hubble

    def tf_nowiggles(self, k_hmpc):
        """
        Parameters
        ----------
        :param k_hmpc: Wavenumber in units of (h Mpc^-1)
        :return: The value of an approximate transfer function that captures the
        non-oscillatory part of a partial baryon transfer function.  In other words,
        the baryon oscillations are left out, but the suppression of power below
        the sound horizon is included.
        """

        k = k_hmpc * self.hubble;    # Convert to Mpc^-1 #
        omhh = omega0 * hubble * hubble

        q = k / 13.41 / self.k_equality
        xx = k * tf_sound_horizon_fit(omhh,f_baryon,1.0)

        alpha_gamma = 1.0-0.328*np.log(431.0*omhh) * f_baryon + 0.38 * np.log(22.3*omhh) * f_baryon**2
        gamma_eff = omhh * (alpha_gamma+(1.0-alpha_gamma) / (1.0+(0.43*xx)**4))
        q_eff = q * omhh / gamma_eff

        T_nowiggles_L0 = np.log(2.0*np.e+1.8*q_eff)
        T_nowiggles_C0 = 14.2 + 731.0/(1+62.5*q_eff)
        return T_nowiggles_L0/(T_nowiggles_L0+T_nowiggles_C0*q_eff**2)

    # ======================= Zero Baryon Formula =========================== #

    def tf_zerobaryon(self, k_hmpc):
        """
        Parameters
        ----------
        :param k_hmpc: Wavenumber in units of (h Mpc^-1)
        :return: The value of the transfer function for a zero-baryon universe.
        """
        k = k_hmpc * self.hubble    # Convert to Mpc^-1 #
        q = k / 13.41 / self.k_equality

        T_0_L0 = np.log(2.0*np.e+1.8*q)
        T_0_C0 = 14.2 + 731.0/(1+62.5*q)
        return T_0_L0/(T_0_L0+T_0_C0*q*q)

    


