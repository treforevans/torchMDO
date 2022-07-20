import torch
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import matplotlib.pyplot as plt
from torchmdo.fea.beam import BeamFEA

Tensor = torch.Tensor


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
        torch.manual_seed(0)
        EI = max(torch.rand(()), 1e-4)  # type: ignore
        L = 0.9
        F = 1.2
        fea = BeamFEA(
            lengths=Tensor([L]),
            EIs=Tensor([EI]),
            fixed_rotation=torch.as_tensor([True, False,], dtype=torch.bool),
            fixed_translation=torch.as_tensor([True, False,], dtype=torch.bool),
        )
        displacements = fea.get_displacements(f=torch.Tensor([[0, 0], [F, 0]]))
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
        torch.manual_seed(0)
        EI = max(torch.rand(()), 1e-4)  # type: ignore
        L = 0.9
        F = 1.2
        n_elems = 100
        fea = BeamFEA(
            lengths=Tensor([L / n_elems,] * n_elems),
            EIs=Tensor([EI,] * n_elems),
            fixed_rotation=torch.as_tensor(
                [True,] + [False,] * n_elems, dtype=torch.bool
            ),
            fixed_translation=torch.as_tensor(
                [True,] + [False,] * n_elems, dtype=torch.bool
            ),
        )
        displacements = fea.get_displacements(
            f=torch.Tensor([[0, 0],] * n_elems + [[F, 0],])  # type: ignore
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
            plt.plot(torch.linspace(0, 1, steps=n_elems + 1), -displacements[:, 0], "b")
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
        torch.manual_seed(0)
        EI = max(torch.rand(()), 1e-4)  # type: ignore
        L = 0.9
        q = 1.2
        fea = BeamFEA(
            lengths=Tensor([L]),
            EIs=Tensor([EI]),
            fixed_rotation=torch.as_tensor([True, False,], dtype=torch.bool),
            fixed_translation=torch.as_tensor([True, False,], dtype=torch.bool),
        )
        displacements = fea.get_displacements(uniform_loads=Tensor([q,]))
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
        torch.manual_seed(0)
        EI = max(torch.rand(()), 1e-4)  # type: ignore
        L = 0.9
        q = 1.2
        n_elems = 100
        fea = BeamFEA(
            lengths=Tensor([L / n_elems,] * n_elems),
            EIs=Tensor([EI,] * n_elems),
            fixed_rotation=torch.as_tensor(
                [True,] + [False,] * n_elems, dtype=torch.bool
            ),
            fixed_translation=torch.as_tensor(
                [True,] + [False,] * n_elems, dtype=torch.bool
            ),
        )
        displacements = fea.get_displacements(uniform_loads=Tensor([q,] * n_elems))
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
            plt.plot(torch.linspace(0, 1, steps=n_elems + 1), -displacements[:, 0], "b")
            plt.plot([0, 1], [0, 0], "k")
            plt.ylabel("deflected position")


if __name__ == "__main__":
    # run the tests with figures to plot
    TestEndLoadedCantileveredBeam().test_multiple_beam_elements(to_plot=True)
    TestUniformlyLoadedCantileveredBeam().test_multiple_beam_elements(to_plot=True)
