import torchmdo.interp as interp
import math
import torch
import gpytorch
import numpy as np
import scipy
from pdb import set_trace


def test_interp():
    """
    Make sure C0 continuous interp agrees with scipy's linear interpolation.
    """
    # generate training data
    x = torch.linspace(0, 1, 40).reshape((-1, 1))*100
    y = torch.hstack(
        [torch.sin(x/100 * 10 * (2 * math.pi)), torch.cos(x/100 * 10 * (2 * math.pi)),]
    )
    # generate test points
    test_x = torch.linspace(0, 1, 501).reshape((-1, 1))*100
    # linearly interp with scipy
    lin_interp = scipy.interpolate.interp1d(x=x[:, 0], y=y, kind="linear", axis=0)
    # now test
    interp_test = interp.Interp1D(x=x, y=y, differentiability=1)
    np.testing.assert_array_almost_equal(
        lin_interp(test_x[:, 0]), interp_test(test_x).detach(), decimal=2.9
    )
    return lin_interp, interp_test, test_x

if __name__ == "__main__":
    test_interp()