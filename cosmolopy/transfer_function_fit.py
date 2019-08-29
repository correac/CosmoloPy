"""
The following routines implement all of the fitting formulae in
Eisenstein \& Hu (1998).

There are seven functions in the class. __init()__ sets the cosmology.
tf_fit_k_hmpc() and tf_fit_k_mpc() calculate the transfer function for an
arbitrary CDM+baryon universe using the fitting formula in Section 3 of the
paper: Eisenstein \& Hu (1998) ApJ, Vol. 496, Issue 2, pp. 605-614.

The functions tf_sound_horizon_fit(), tf_k_peak(), tf_nowiggles(), and
tf_zerobaryon() calculate other quantities given in Section 4 of the paper.

Please note that while the routines use Mpc^-1 units internally, this driver
has been written to take an array of wavenumbers in units of h Mpc^-1. On the
other hand, if you want to use Mpc^-1 externally, you can do this by
altering the variables you pass to the driver: omega0 -> omega0*hubble*hubble,
hubble -> 1.0."""

import numpy as np

class TransferFunctionfit:

    def __init__(self, omega0, hubble, f_baryon, Tcmb):
        """
        Parameters
        ----------
        omega0: scalar
            The density of CDM and baryons, in units of critical dens
        hubble: scalar
            Hubble constant, in units of 100 km/s/Mpc
        f_baryon: scalar
            Fraction of baryons to CDM
        Tcmb: scalar
            The temperature of the CMB in Kelvin. Tcmb<=0 forces use of the COBE value of 2.728 K.
            
        Note
        ----
        Units are always Mpc, never h^-1 Mpc.
        """

        self.hubble = hubble
        self.omhh = omega0 * self.hubble**2
        self.f_baryon = f_baryon
        self.obhh = self.omhh * self.f_baryon
        if Tcmb<=0.0: Tcmb=2.728
        theta_cmb = Tcmb/2.7

        z_equality = 2.50e4 * self.omhh / theta_cmb**4  # Really 1+z #
        self.k_equality = 0.0746 * self.omhh / theta_cmb**2

        z_drag_b1 = 0.313 * self.omhh**(-0.419) * (1.0+0.607 * self.omhh**0.674)
        z_drag_b2 = 0.238 * self.omhh**0.223
        z_drag = 1291. * self.omhh**0.251 / (1.0+0.659 * self.omhh**0.828) * (1.0+z_drag_b1 * self.obhh**z_drag_b2)

        R_drag = 31.5 * self.obhh / theta_cmb**4 * (1000./(1.0+z_drag))
        R_equality = 31.5 * self.obhh / theta_cmb**4 * (1000./z_equality)

        self.sound_horizon = 2. /3. / self.k_equality * np.sqrt(6./R_equality)
        self.sound_horizon *= np.log((np.sqrt(1.0+R_drag)+np.sqrt(R_drag+R_equality))/(1.0+sqrt(R_equality)))

        self.k_silk = 1.6 * self.obhh**0.52 * self.omhh**0.73 * (1.0+(10.4 * self.omhh)**(-0.95))

        alpha_c_a1 = (46.9*self.omhh)**0.670 * (1.0+(32.1*self.omhh)**(-0.532))
        alpha_c_a2 = (12.0*self.omhh)**0.424 * (1.0+(45.0*self.omhh)**(-0.582))
        self.alpha_c = alpha_c_a1**(-self.f_baryon) * alpha_c_a2**(-self.f_baryon**3)

        beta_c_b1 = 0.944 / (1.0 + (458.*self.omhh)**(-0.708))
        beta_c_b2 =  (0.395 * self.omhh)**(-0.0266)
        self.beta_c = 1.0 / (1.0+beta_c_b1 * ((1.0-self.f_baryon)**beta_c_b2)-1.0)

        y = z_equality / (1.0 + z_drag)
        alpha_b_G = y * (-6.*np.sqrt(1.0+y)+(2.+3.*y)*np.log((np.sqrt(1.0+y)+1.0)/(np.sqrt(1.0+y)-1.0)))
        self.alpha_b = 2.07 * self.k_equality * self.sound_horizon * (1.0+R_drag)**(-0.75) * alpha_b_G

        self.beta_node = 8.41 * self.omhh**0.435
        self.beta_b = 0.5+self.f_baryon+(3.-2.*self.f_baryon) * np.sqrt((17.2 * self.omhh)**2+1.0)

        self.sound_horizon_fit = 44.5 * np.log(9.83/self.omhh)/np.sqrt(1.0+10.0 * self.obhh**0.75)
        k_peak = 2.5 * np.pi * (1.0+0.217 * self.omhh) / self.sound_horizon #sound_horizon_fit I think!#

        self.alpha_gamma = 1.0-0.328 * np.log(431.0 * self.omhh) * self.f_baryon
        self.alpha_gamma += 0.38 * np.log(22.3 * self.omhh) * (self.f_baryon)**2
    
    def tf_fit_k_mpc(self,k):
        """
        Parameters
        ----------
        k : array
            Wavenumber in Mpc^-1.

        Returns
        -------
        Returns the value of the full transfer function fitting formula.
        This is the form given in Section 3 of Eisenstein & Hu (1998).
        
        tf_full : array
            The full fitting formula, eq. (16), for the matter transfer function.
        tf_baryon : array
            The baryonic piece of the full fitting formula, eq. (21).
        tf_cdm : array
            The CDM piece of the full fitting formula, eq. (17). """

        # Notes: Units are Mpc, not h^-1 Mpc. #

        k = np.abs(k)    # Just define negative k as positive #

        q = k / 13.41 / self.k_equality
        xx = k * self.sound_horizon

        T_c_ln_beta = np.log(np.e+1.8 * self.beta_c * q)
        T_c_ln_nobeta = np.log(np.e+1.8*q)
        T_c_C_alpha = 14.2 / self.alpha_c + 386.0 / (1.0+69.9 * q**1.08)
        T_c_C_noalpha = 14.2 + 386.0/(1.0+69.9 * q**1.08)

        T_c_tilde = T_c_ln_beta / (T_c_ln_beta+T_c_C_noalpha * q**2)
        T_c_f = 1.0/(1.0+(xx/5.4)**4))
        T_c = T_c_f * T_c_tilde + (1.0-T_c_f) * T_c_tilde

        s_tilde = self.sound_horizon / (1.0+(self.beta_node / xx)**3)**(1./3.)
        xx_tilde = k * s_tilde

        T_b = np.sin(xx_tilde) / (xx_tilde)
        T_b *= (T_c_tilde/(1.0+(xx/5.2)**2) + self.alpha_b / (1.0+(self.beta_b/xx)**3) * np.exp(-1.0 * (k/self.k_silk)**1.4))

        T_full = self.f_baryon * T_b + (1.0-self.f_baryon) * T_c

        return T_full, T_b, T_c #Full, baryon, CDM#
    
    def tf_fit_k_hmpc(self, k):
        """
        Parameters
        ----------
        k : array
            Wavenumber in h Mpc^-1.
        
        Returns
        -------
        The value of the full transfer function fitting formula.
        
        tf_full : array
            The full fitting formula, eq. (16), for the matter transfer function.
        tf_baryon : array
            The baryonic piece of the full fitting formula, eq. (21).
        tf_cdm : array
            The CDM piece of the full fitting formula, eq. (17).
        """
        return self.tf_fit_k_mpc(k * self.hubble)


    # ======================= Approximate forms =========================== #

    def tf_sound_horizon_fit(self, h=None):
        """
        Parameters
        ----------
        h : scalar
            Hubble constant, in units of 100 km/s/Mpc.
            
        Returns
        -------
        The approximate value of the sound horizon, in h^-1 Mpc.

        Note
        ----
        If you prefer to have the answer in  units of Mpc, use h -> 1.
        Otherwise do not specify h.
        """
        if h==None:
            return self.sound_horizon_fit * self.hubble
        else:
            om_hh = self.omega0 * h * h
            ob_hh = om_hh * self.f_baryon
            s_fit = 44.5 * np.log(9.83/om_hh)/np.sqrt(1.0+10.0 * ob_hh**0.75)
            return s_fit * h

    def tf_k_peak(self):
        """
        Returns
        -------
        The approximate location of the first baryonic peak, in h Mpc^-1
        """
        k_peak_mpc = 2.5 * np.pi * (1.0+0.217*self.omhh) / self.tf_sound_horizon_fit(h=1.0)
        return k_peak_mpc / self.hubble

    def tf_nowiggles(self, k_hmpc):
        """
        Parameters
        ----------
        k_hmpc : array
            Wavenumber in h Mpc^-1.
            
        Returns
        -------
        The value of an approximate transfer function that captures the
        non-oscillatory part of a partial baryon transfer function.  In other words,
        the baryon oscillations are left out, but the suppression of power below
        the sound horizon is included.
        """

        k = k_hmpc * self.hubble;    # Convert to Mpc^-1 #
        q = k / 13.41 / self.k_equality
        xx = k * self.tf_sound_horizon_fit(h=1.0)

        gamma_eff = self.omhh * (self.alpha_gamma+(1.0-self.alpha_gamma) / (1.0+(0.43*xx)**4))
        q_eff = q * self.omhh / gamma_eff

        T_nowiggles_L0 = np.log(2.0 * np.e + 1.8 * q_eff)
        T_nowiggles_C0 = 14.2 + 731.0 / (1.0 + 62.5 * q_eff)
        return T_nowiggles_L0/(T_nowiggles_L0+T_nowiggles_C0 * q_eff**2)

    # ======================= Zero Baryon Formula =========================== #

    def tf_zerobaryon(self, k_hmpc):
        """
        Parameters
        ----------
        k_hmpc : array
            Wavenumber in h Mpc^-1.
            
        Returns
        -------
        The value of the transfer function for a zero-baryon universe.
        """
        k = k_hmpc * self.hubble    # Convert to Mpc^-1 #
        q = k / 13.41 / self.k_equality

        T_0_L0 = np.log(2.0*np.e+1.8*q)
        T_0_C0 = 14.2 + 731.0/(1+62.5*q)
        return T_0_L0/(T_0_L0+T_0_C0*q*q)


