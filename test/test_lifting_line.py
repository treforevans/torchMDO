import torch
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import matplotlib.pyplot as plt
from torchmdo.lifting_line.wing import LiftingLineWing
from pdb import set_trace


class TestLiftingLine:
    def test_square_wing(self, to_plot=False):
        # Generate a square, untapered wing
        b = 33.6
        # AR = 10; c = b/AR
        c = 0.756
        AR = b ** 2 / (b * c)
        # establish a computational grid (using the lifting line change of variables)
        spanwise_loc = (
            -b / 2 * torch.cos(torch.linspace(torch.pi / 2, torch.pi, steps=101))
        )
        section_lengths = torch.diff(spanwise_loc)
        spanwise_loc = spanwise_loc[1:]  # don't include the root location
        assert torch.abs(spanwise_loc[-1] - b / 2) < 1e-6
        chords = torch.zeros_like(spanwise_loc) + c
        alpha_sections = torch.zeros_like(spanwise_loc) + 0.5 * torch.pi / 180
        alpha_0s = torch.zeros_like(spanwise_loc) - 0.3 * torch.pi / 180
        wing_area = b * c
        assert_almost_equal(2 * torch.sum(chords * section_lengths), wing_area)
        wing = LiftingLineWing(
            spanwise_loc, chords, alpha_sections, alpha_0s, span=b, wing_area=wing_area
        )
        print("aspect ratio:", AR)

        # Compute lift distribution for given CL
        Cl_target = 1.2 * torch.ones(())
        Cl_section = wing.section_lift_coeff(Cl=Cl_target)
        if to_plot:
            plt.figure()
            plt.plot(spanwise_loc, Cl_section)
            plt.xlabel("spanwise location")
            plt.ylabel("Section Cl")
        print("Avg section Cl:", Cl_section.mean())

        # Compute `alpha_effective`
        if to_plot:
            plt.figure()
            plt.plot(spanwise_loc, wing.alpha_effective(Cl=Cl_target) * 180 / torch.pi)
            plt.xlabel("spanwise location")
            plt.ylabel("effective AoA")

        # Find the aircraft Angle of Attack
        print("Aircraft AOA:", wing.alpha_aircraft * 180 / torch.pi)

        # Compute the induced drag and span efficiency factor
        Cdi, span_efficiency_factor, induced_drag_factor = wing.induced_drag(
            Cl=Cl_target
        )
        print("Induced drag coeff:",)
        print("Induced drag factor:", induced_drag_factor)
        print("Span efficiency factor:", span_efficiency_factor)

        # Check `A1`
        assert_almost_equal(Cl_target, wing.A1 * torch.pi * AR)

        # Check that the Cl of the sections result in the correct target Cl
        assert_almost_equal(
            Cl_target,
            torch.sum(chords * section_lengths * wing.section_lift_coeff(Cl=Cl_target))
            / (0.5 * wing_area),
            err_msg="Cl target was not met",
            decimal=2,
        )

        # Verify that the change of variables are correct
        assert_array_almost_equal(
            spanwise_loc, -b / 2 * torch.cos(wing.spanwise_thetas)
        )

        # Check that Prandtl's equations are satisfied properly (Anderson eq 5.51)
        alphas = wing.alpha_sections + wing.alpha_aircraft
        ns = torch.cat([torch.Tensor([1,]), wing.ns])
        Ans = torch.cat([torch.Tensor([wing.A1,]), wing.Ans])
        assert ns.shape == Ans.shape
        for alpha, alpha_0, chord, theta in zip(
            alphas, alpha_0s, chords, wing.spanwise_thetas
        ):
            alpha_test = (
                2.0 * b / (torch.pi * chord) * torch.sum(Ans * torch.sin(ns * theta))
                + alpha_0s
                + torch.sum(ns * Ans * torch.sin(ns * theta) / torch.sin(theta))
            )
            assert_almost_equal(alpha.numpy(), alpha_test.numpy())


"""
For further validation, consider reproduction of figure 5.20 from Anderson's textbook
which plots the induced drag factor as a function of linear taper ratio for several
aspect ratio wings.
"""


if __name__ == "__main__":
    # run the tests with figures to plot
    TestLiftingLine().test_square_wing(to_plot=True)
