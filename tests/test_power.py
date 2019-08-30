"""Check that the power module is working."""

#import numpy.testing.utils as ntest
import cosmolopy.transfer_function
cosmo = {}
cosmo['omega_b_0'] = 0.04
cosmo['omega_M_0'] = 0.3
cosmo['omega_lambda_0'] = 0.7
cosmo['h'] = 0.72
cosmo['omega_n_0'] = 0.0
cosmo['N_nu'] = 0.0
redshift = 0.0

TF = cosmolopy.transfer_function.TransferFunction(redshift,**cosmo)
tf_cb, tf_cbnu = TF.tf_k_hmpc(100.0)
print('Obtained:',tf_cb)
print('Expected:',1.48236347286e-06)
#power.TFmdm_set_cosm(0.3, 0.04, 0.0, 0, 0.7, 0.72, 0.0)
#ntest.assert_approx_equal(tf_cb,1.48236347286e-06)
