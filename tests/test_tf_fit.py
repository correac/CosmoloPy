"""Check that the tf_fit module is working."""

#import numpy.testing.utils as ntest
import cosmolopy.transfer_function_fit
omhh = 0.136
h = 0.7
omega_m = omhh/h**2
TF = cosmolopy.transfer_function_fit.TransferFunctionfit(omega_m, h, 0.2, 2.728)
tf_full, tf_b, tf_cdm = TF.tf_fit_k_mpc(10.0)
print("Obtained:",tf_full)
print("Expected:",3.69328954548e-05)
#ntest.assert_approx_equal(tf_full,3.69328954548e-05)
