import numpy as np
from scipy.linalg import lu_factor, lu_solve
from pdb import set_trace


class LiftingLineWing:
    def __init__(self, spanwise_loc, chords, alpha_sections, alpha_0s, AR):
        """
        Inputs:
            spanwise_loc : (m,) spanwise locations of the m wing segments. The root is 
                considered to be zero and the node is the half-wing span.
                We assume symmetrical loading so this should just be points from one half of wing.
            chords : (m,) chord of each wing segment
            alpha_sections : (m,) AoA (in rad) of each wing segment assuming zero aircraft AoA
            alpha_0s : (m,) zero lift AoA (in rad) of each wing segment assuming zero aircraft AoA
            AR : aspect ratio (b^2/S)
        """
        # internalize variables
        self.m = np.size(spanwise_loc)
        self.spanwise_loc = np.atleast_1d(spanwise_loc)
        self.chords = np.atleast_1d(chords)
        self.alpha_sections = np.atleast_1d(alpha_sections)
        self.alpha_0s = np.atleast_1d(alpha_0s)
        self.AR = AR
        assert (
            self.spanwise_loc.shape
            == self.chords.shape
            == self.alpha_sections.shape
            == self.alpha_0s.shape
        )
        assert np.all(
            self.spanwise_loc > 0
        ), "should not put a spanwise loc at the root"
        assert np.all(
            self.spanwise_loc == np.unique(self.spanwise_loc)
        ), "spanwise loc should be unique and sorted"
        assert np.all(self.chords > 0)
        assert np.all(self.alpha_0s <= 0), "assuming lifting airfoils, check input"
        self.span = 2.0 * self.spanwise_loc[-1]
        # perform change of variables
        self.spanwise_thetas = np.arccos(-self.spanwise_loc * 2.0 / self.span)
        assert (
            self.spanwise_thetas.min() > np.pi / 2
            and self.spanwise_thetas.max() <= np.pi
        ), "sanity check"
        # setup and factorize linear system Ax = b (notebook Apr 24, 2020)
        # include only odd n values. Also I want one less coefficient than m since
        # aircraft AoA unknown.
        self.ns = np.arange(start=3, stop=self.m * 2 + 1, step=2)
        ns = self.ns.reshape((1, -1))  # reshape for broadcasting
        thetas = self.spanwise_thetas.reshape((-1, 1))  # reshape for broadcasting
        self.A_sys = np.zeros((self.m,) * 2)
        self.A_sys[:, 0] = -1.0  # contribution of aircraft AoA
        self.A_sys[:, 1:] = np.sin(ns * thetas) * (
            2.0 * self.span / (np.pi * self.chords.reshape((-1, 1)))
            + ns / np.sin(thetas)
        )
        self.lu_and_piv = lu_factor(self.A_sys)

    def solve(self, Cl):
        """
        solve lifting line equations given a target aircraft Cl
        """
        self.Cl = Cl
        # given Cl, compute A1
        self.A1 = self.Cl / (np.pi * self.AR)
        # compute the target vector b to solve Ax=b
        b_sys = (
            self.alpha_sections
            - 2.0
            * self.span
            * self.A1
            * np.sin(self.spanwise_thetas)
            / (np.pi * self.chords)
            - self.alpha_0s
            - self.A1
        ).reshape((-1, 1))
        # solve the system
        x = lu_solve(self.lu_and_piv, b_sys)
        self.alpha_aircraft = np.squeeze(x[0])
        self.Ans = np.squeeze(x[1:])  # coefficients corresponding to ns

    def induced_drag_coeff(self):
        """ compute the induced drag coefficient """
        # compute induced drag factor and span efficiency factor
        self.induced_drag_factor = np.sum(
            self.ns * np.square(self.Ans / self.A1)
        )  # (delta)
        assert self.induced_drag_factor >= 0
        self.span_efficiency_factor = 1.0 / (1.0 + self.induced_drag_factor)
        self.Cdi = np.square(self.Cl) / (np.pi * self.span_efficiency_factor * self.AR)
        return self.Cdi

    def alpha_induced_fun(self, spanwise_loc):
        """
        Induced AoA at a given spanwise locations.
        Effective angle of attack:
            alpha_effective = alpha_section + alpha_aircraft - alpha_induced
        Local Section Lift (assuming thin airfoil theory):
            cl = 2.*np.pi*(alpha_effective - alpha_0)
        """
        assert np.all(spanwise_loc >= 0) and np.all(spanwise_loc <= self.span / 2)
        thetas = np.arccos(-spanwise_loc * 2.0 / self.span).reshape((1, -1))
        ns = self.ns.reshape((1, -1))  # reshape for broadcasting
        alpha_induced = self.A1 + np.sum(
            ns * self.Ans * np.sin(ns * thetas) / np.sin(thetas), axis=1
        )
        return alpha_induced

    def alpha_induced(self):
        """ compute induced angle of attack at each spanwise reference """
        ns = self.ns.reshape((1, -1))  # reshape for broadcasting
        thetas = self.spanwise_thetas.reshape((-1, 1))  # reshape for broadcasting
        alpha_induced = self.A1 + np.sum(
            ns * self.Ans * np.sin(ns * thetas) / np.sin(thetas), axis=1
        )
        return alpha_induced

    def alpha_effective(self):
        """ compute effective angle of attack at each spanwise reference """
        alpha_induced = self.alpha_induced()
        alpha_effective = self.alpha_sections + self.alpha_aircraft - alpha_induced
        return alpha_effective

    def section_lift_coeff(self):
        """ compute sectional lift coefficient assuming thin airfoil theory """
        alpha_effective = self.alpha_effective()
        cl = 2.0 * np.pi * (alpha_effective - self.alpha_0s)
        return cl

