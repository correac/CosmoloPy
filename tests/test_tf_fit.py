"""Check that the tf_fit module is working."""

import numpy.testing.utils as ntest
import cosmolopy.tranfer_function_fit
#omhh = 0.136#
TF = TransferFunctionfit(0.277, 0.7, 0.2, 2.728)
tf_full, tf_b, tf_cdm = TF.tf_fit_k_mpc(10.0)
ntest.assert_approx_equal(tf_full,3.69328954548e-05)
