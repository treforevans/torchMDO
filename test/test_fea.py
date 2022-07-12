import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import matplotlib.pyplot as plt
from torchmdao.fea.beam import BeamFEA


class TestEndLoadedCantileveredBeam:
    r"""
    End Loaded Cantilevered Beam
    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Cantilever_with_end_load.svg/330px-Cantilever_with_end_load.svg.png)
    From [wiki](https://en.wikipedia.org/wiki/Deflection_(engineering)), the deflection 
    at the tip point load of F with a beam of length L should be:
    $\frac{F L^3}{3EI}$ and the angle of deflection should be $\frac{FL^2}{2 EI}$.
    """

    def test_single_beam_element(self):
        # Let's verify this for a single beam segment:
        EI = max(np.random.rand(), 1e-4)
        L = 0.9
        F = 1.2
        fea = BeamFEA(
            lengths=[L],
            EIs=[EI],
            fixed_rotation=[True, False],
            fixed_translation=[True, False],
        )
        displacements = fea.get_displacements(f=np.array([[0, 0], [F, 0]]))
        assert_almost_equal(
            actual=displacements[1, 0],
            desired=(F * L ** 3) / (3.0 * EI),
            err_msg="displacement wrong",
        )
        assert_almost_equal(
            actual=displacements[1, 1],
            desired=(F * L ** 2) / (2.0 * EI),
            err_msg="angular deflection wrong",
        )

    def test_multiple_beam_elements(self, to_plot=False):
        # Now verify this for a beam consisting of multiple elements with the same EI
        EI = max(np.random.rand(), 1e-4)
        L = 0.9
        F = 1.2
        n_elems = 100
        fea = BeamFEA(
            lengths=[L / n_elems,] * n_elems,
            EIs=[EI,] * n_elems,
            fixed_rotation=[True,] + [False,] * n_elems,
            fixed_translation=[True,] + [False,] * n_elems,
        )
        displacements = fea.get_displacements(
            f=np.array([[0, 0],] * n_elems + [[F, 0],])
        )
        assert_almost_equal(
            actual=displacements[-1, 0],
            desired=(F * L ** 3) / (3.0 * EI),
            err_msg="displacement wrong",
        )
        assert_almost_equal(
            actual=displacements[-1, 1],
            desired=(F * L ** 2) / (2.0 * EI),
            err_msg="angular deflection wrong",
        )

        if to_plot:
            # plot the deflection
            plt.figure()
            plt.plot(np.linspace(0, 1, num=n_elems + 1), -displacements[:, 0], "b")
            plt.plot([0, 1], [0, 0], "k")
            plt.ylabel("deflected position")


class TestUniformlyLoadedCantileveredBeam:
    """
    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Cantilever_with_uniform_distributed_load.svg/2560px-Cantilever_with_uniform_distributed_load.svg.png)
    From [wiki](https://en.wikipedia.org/wiki/Deflection_(engineering)), the 
    deflection at the tip given a uniformly distributed load q should be:
    $\frac{qL^4}{8EI}$ and the angle of deflection should be $\frac{qL^3}{6 EI}$.
    """

    def test_single_beam_element(self):
        # Let's verify this for a single beam segment:
        EI = max(np.random.rand(), 1e-4)
        L = 0.9
        q = 1.2
        fea = BeamFEA(
            lengths=[L],
            EIs=[EI],
            fixed_rotation=[True, False],
            fixed_translation=[True, False],
        )
        displacements = fea.get_displacements(uniform_loads=[q])
        assert_almost_equal(
            actual=displacements[1, 0],
            desired=(q * L ** 4) / (8.0 * EI),
            err_msg="displacement wrong",
        )
        assert_almost_equal(
            actual=displacements[1, 1],
            desired=(q * L ** 3) / (6.0 * EI),
            err_msg="angular deflection wrong",
        )

    def test_multiple_beam_elements(self, to_plot=False):
        # Now verify this for a beam consisting of multiple elements with the same EI

        EI = max(np.random.rand(), 1e-4)
        L = 0.9
        q = 1.2
        n_elems = 100
        fea = BeamFEA(
            lengths=[L / n_elems,] * n_elems,
            EIs=[EI,] * n_elems,
            fixed_rotation=[True,] + [False,] * n_elems,
            fixed_translation=[True,] + [False,] * n_elems,
        )
        displacements = fea.get_displacements(uniform_loads=[q,] * n_elems)
        assert_almost_equal(
            actual=displacements[-1, 0],
            desired=(q * L ** 4) / (8.0 * EI),
            err_msg="displacement wrong",
        )
        assert_almost_equal(
            actual=displacements[-1, 1],
            desired=(q * L ** 3) / (6.0 * EI),
            err_msg="angular deflection wrong",
        )

        # plot the deflection
        if to_plot:
            plt.figure()
            plt.plot(np.linspace(0, 1, num=n_elems + 1), -displacements[:, 0], "b")
            plt.plot([0, 1], [0, 0], "k")
            plt.ylabel("deflected position")

    def test_max_strain(self, to_plot=False):
        """ informal test where the max strain is evaluated and plotted. """
        EI = max(np.random.rand(), 1e-4)
        L = 0.9
        q = 1.2
        n_elems = 100

        fea = BeamFEA(
            lengths=[L / n_elems,] * n_elems,
            EIs=[EI,] * n_elems,
            fixed_rotation=[True,] + [False,] * n_elems,
            fixed_translation=[True,] + [False,] * n_elems,
            thicknesses=np.linspace(0.05, 0.02, n_elems),
        )
        displacements = fea.get_displacements(uniform_loads=[q,] * n_elems)
        maxStrain = fea.get_max_strain(displacements)

        # plot the deflection and strain
        if to_plot:
            plt.figure()
            plt.plot(np.linspace(0, 1, num=n_elems + 1), -displacements[:, 0], "b")
            plt.plot([0, 1], [0, 0], "k")
            plt.plot(np.linspace(0, 1, num=n_elems), -maxStrain, "r")
            plt.ylabel("deflected position and max strain")


if __name__ == "__main__":
    # run the tests with figures to plot
    TestEndLoadedCantileveredBeam().test_multiple_beam_elements(to_plot=True)
    TestUniformlyLoadedCantileveredBeam().test_multiple_beam_elements(to_plot=True)
    TestUniformlyLoadedCantileveredBeam().test_max_strain(to_plot=True)
