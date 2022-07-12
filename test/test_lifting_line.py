import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import matplotlib.pyplot as plt
from torchmdao.lifting_line.wing import LiftingLineWing


class TestLiftingLine:
    def test_square_wing(self, to_plot=False):
        # Generate a square, untapered wing
        b = 33.6
        # AR = 10; c = b/AR
        c = 0.756
        AR = b ** 2 / (b * c)
        spanwise_loc = np.linspace(0, b / 2, num=101)[
            1:
        ]  # don't include the root location
        chords = np.zeros_like(spanwise_loc) + c
        alpha_sections = np.zeros_like(spanwise_loc) + 0.5 * np.pi / 180
        alpha_0s = np.zeros_like(spanwise_loc) - 0.3 * np.pi / 180
        wing = LiftingLineWing(spanwise_loc, chords, alpha_sections, alpha_0s, AR)
        print("aspect ratio:", AR)

        # Compute lift distribution for given CL
        Cl_target = 1.0
        wing.solve(Cl=Cl_target)
        Cl_section = wing.section_lift_coeff()
        if to_plot:
            plt.figure()
            plt.plot(spanwise_loc, Cl_section)
            plt.xlabel("spanwise location")
            plt.ylabel("Section Cl")
        print("Avg section Cl:", Cl_section.mean())

        # Compute `alpha_effective`
        if to_plot:
            plt.figure()
            plt.plot(spanwise_loc, wing.alpha_effective() * 180 / np.pi)
            plt.xlabel("spanwise location")
            plt.ylabel("effective AoA")

        # Find the aircraft Angle of Attack
        print("Aircraft AOA:", wing.alpha_aircraft * 180 / np.pi)

        # Compute the induced drag and span efficiency factor
        print("Induced drag coeff:", wing.induced_drag_coeff())
        print("Induced drag factor:", wing.induced_drag_factor)
        print("Span efficiency factor:", wing.span_efficiency_factor)

        # Check `A1`
        assert_almost_equal(Cl_target, wing.A1 * np.pi * AR)

        # Verify that the change of variables are correct
        assert_array_almost_equal(spanwise_loc, -b / 2 * np.cos(wing.spanwise_thetas))

        # Check that Prandtl's equations are satisfied properly (Anderson eq 5.51)
        alphas = wing.alpha_sections + wing.alpha_aircraft
        ns = np.concatenate([[1,], wing.ns])
        Ans = np.concatenate([[wing.A1,], wing.Ans])
        assert ns.shape == Ans.shape
        for alpha, alpha_0, chord, theta in zip(
            alphas, alpha_0s, chords, wing.spanwise_thetas
        ):
            assert_almost_equal(
                alpha,
                2.0 * b / (np.pi * chord) * np.sum(Ans * np.sin(ns * theta))
                + alpha_0s
                + np.sum(ns * Ans * np.sin(ns * theta) / np.sin(theta)),
            )


"""
For further validation, consider reproduction of figure 5.20 from Anderson's textbook
which plots the induced drag factor as a function of linear taper ratio for several
aspect ratio wings.
"""


if __name__ == "__main__":
    # run the tests with figures to plot
    TestLiftingLine().test_square_wing(to_plot=True)