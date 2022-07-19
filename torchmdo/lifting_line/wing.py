import torch
from typing import Tuple
from pdb import set_trace

Tensor = torch.Tensor


class LiftingLineWing:
    """
    Lifting-line model of an aircraft wing.
    This is used to compute the lift distribution and induced drag over a straight wing.
    Note that only the specifications for half of the wing should be provided.

    Args:
        spanwise_loc : shape (m,) spanwise locations of the m wing segments. The root is 
            considered to be zero and the node is the half-wing span.
            We assume symmetrical loading so this should just be points 
            from one half of wing.
        chords : shape (m,) chord of each wing segment
        alpha_sections : shape (m,) AoA (in rad) of each wing segment assuming zero 
            aircraft AoA.
        alpha_0s : shape (m,) zero lift AoA (in rad) of each wing segment assuming zero 
            aircraft AoA.
        AR : aspect ratio of the wing given by :math:`b^2/S` where `b` is the span and 
            `S` is the wing area.
    """

    def __init__(
        self,
        spanwise_loc: Tensor,
        chords: Tensor,
        alpha_sections: Tensor,
        alpha_0s: Tensor,
        AR: Tensor,
    ):
        # internalize variables
        self.m = torch.numel(spanwise_loc)
        self.spanwise_loc = torch.atleast_1d(spanwise_loc)
        self.chords = torch.atleast_1d(chords)
        self.alpha_sections = torch.atleast_1d(alpha_sections)
        self.alpha_0s = torch.atleast_1d(alpha_0s)
        self.AR = AR
        assert (
            self.spanwise_loc.shape
            == self.chords.shape
            == self.alpha_sections.shape
            == self.alpha_0s.shape
        )
        assert torch.all(
            self.spanwise_loc > 0
        ), "should not put a spanwise loc at the root"
        assert torch.all(
            self.spanwise_loc == torch.unique(self.spanwise_loc)
        ), "spanwise loc should be unique and sorted"
        assert torch.all(self.chords > 0)
        assert torch.all(self.alpha_0s <= 0), "assuming lifting airfoils, check input"
        self.span = 2.0 * self.spanwise_loc[-1]
        # perform change of variables
        self.spanwise_thetas = torch.arccos(-self.spanwise_loc * 2.0 / self.span)
        assert (
            self.spanwise_thetas.min() > torch.pi / 2
            and self.spanwise_thetas.max() <= torch.pi
        ), "sanity check"
        # setup and factorize linear system `Ax = b`` (notebook Apr 24, 2020)
        # include only odd n values. Also I want one less coefficient than m since
        # aircraft AoA unknown.
        self.ns = torch.arange(start=3, end=self.m * 2 + 1, step=2)
        ns = self.ns.reshape((1, -1))  # reshape for broadcasting
        thetas = self.spanwise_thetas.reshape((-1, 1))  # reshape for broadcasting
        # self.A_sys = torch.hstack( # DEBUG
        #     [
        #         -torch.ones((self.m, 1)),  # contribution of aircraft AoA
        #         torch.sin(ns * thetas)
        #         * (
        #             2.0 * self.span / (torch.pi * self.chords.reshape((-1, 1)))
        #             + ns / torch.sin(thetas)
        #         ),
        #     ]
        # )
        self.A_sys = torch.zeros((self.m,) * 2)
        self.A_sys[:, 0] = -1.0  # contribution of aircraft AoA
        self.A_sys[:, 1:] = torch.sin(ns * thetas) * (
            2.0 * self.span / (torch.pi * self.chords.reshape((-1, 1)))
            + ns / torch.sin(thetas)
        )
        self.A_sys_lu = torch.lu(self.A_sys)
        # save the target Cl from the last solve (init to None)
        self.Cl_solved = None

    def solve(self, Cl: Tensor) -> None:
        """
        solve lifting line equations given a target aircraft lift coefficient (:math:`C_L`).

        Args:
            Cl : Target wing lift coefficient (:math:`C_L`).
        """
        assert Cl.numel() == 1
        # determine if the computation has already been completed
        if self.Cl_solved is not None and Cl == self.Cl_solved:
            # this Cl has already been solved so return
            return
        # given Cl, compute A1
        self.A1 = Cl / (torch.pi * self.AR)
        # compute the target vector b to solve Ax=b
        b_sys = (
            self.alpha_sections
            - (
                (2.0 * self.span * self.A1 * torch.sin(self.spanwise_thetas))
                / (torch.pi * self.chords)
            )
            - self.alpha_0s
            - self.A1
        ).reshape((-1, 1))
        # solve the system
        x = torch.lu_solve(b_sys, *self.A_sys_lu)
        self.alpha_aircraft = x[0].reshape(())
        self.Ans = torch.squeeze(x[1:], dim=1)  # coefficients corresponding to ns
        # save the target Cl solved
        self.Cl_solved = Cl.clone()

    def induced_drag(self, Cl: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the wing induced drag coefficient (:math:`C_D`), span efficiency factor, and
        induced drag factor.
        
        Args:
            Cl : Target wing lift coefficient (:math:`C_L`).
        
        Returns:
            ``[induced_drag_coefficient, span_efficiency_factor, induced_drag_factor]``
        """
        self.solve(Cl=Cl)
        # compute induced drag factor and span efficiency factor (delta)
        induced_drag_factor = torch.sum(self.ns * torch.square(self.Ans / self.A1))
        assert induced_drag_factor >= 0, "sanity check"
        span_efficiency_factor = 1.0 / (1.0 + induced_drag_factor)
        induced_drag_coefficient = torch.square(Cl) / (
            torch.pi * span_efficiency_factor * self.AR
        )
        return induced_drag_coefficient, span_efficiency_factor, induced_drag_factor

    def alpha_induced_fun(self, spanwise_loc: Tensor, Cl: Tensor) -> Tensor:
        """
        Induced angle of attack (AoA) at a given spanwise locations.

        Args:
            spanwise_loc: the spanwise locations to evaluate the induced AoA.
            Cl : Target wing lift coefficient (:math:`C_L`).
        Notes:
            To compute effective angle of attack and local section 
            lift (assuming thin airfoil theory):

            >>> alpha_effective = alpha_section + alpha_aircraft - alpha_induced
            >>> cl = 2.*torch.pi*(alpha_effective - alpha_0)
        """
        self.solve(Cl=Cl)
        assert torch.all(spanwise_loc >= 0) and torch.all(spanwise_loc <= self.span / 2)
        thetas = torch.arccos(-spanwise_loc * 2.0 / self.span).reshape((1, -1))
        ns = self.ns.reshape((1, -1))  # reshape for broadcasting
        alpha_induced = self.A1 + torch.sum(
            ns * self.Ans * torch.sin(ns * thetas) / torch.sin(thetas), dim=1
        )
        return alpha_induced

    def alpha_induced(self, Cl: Tensor) -> Tensor:
        """
        compute induced angle of attack at each spanwise reference.
        
        Args:
            Cl : Target wing lift coefficient (:math:`C_L`).
        """
        self.solve(Cl=Cl)
        ns = self.ns.reshape((1, -1))  # reshape for broadcasting
        thetas = self.spanwise_thetas.reshape((-1, 1))  # reshape for broadcasting
        alpha_induced = self.A1 + torch.sum(
            ns * self.Ans * torch.sin(ns * thetas) / torch.sin(thetas), dim=1
        )
        return alpha_induced

    def alpha_effective(self, Cl: Tensor) -> Tensor:
        """
        Compute effective angle of attack at each spanwise reference.
        
        Args:
            Cl : Target wing lift coefficient (:math:`C_L`).
        """
        self.solve(Cl=Cl)
        alpha_induced = self.alpha_induced(Cl=Cl)
        alpha_effective = self.alpha_sections + self.alpha_aircraft - alpha_induced
        return alpha_effective

    def section_lift_coeff(self, Cl: Tensor) -> Tensor:
        """
        Compute the sectional lift coefficient at each spanwise reference assuming 
        thin airfoil theory.
        
        Args:
            Cl : Target wing lift coefficient (:math:`C_L`).
        """
        self.solve(Cl=Cl)
        alpha_effective = self.alpha_effective(Cl=Cl)
        cl = 2.0 * torch.pi * (alpha_effective - self.alpha_0s)
        return cl
