import torch
from .base import BaseFEA
from typing import Optional
from pdb import set_trace

Tensor = torch.Tensor


class BeamFEA(BaseFEA):
    def __init__(
        self,
        lengths: Tensor,
        EIs: Tensor,
        fixed_rotation: Tensor,
        fixed_translation: Tensor,
        thicknesses: Optional[Tensor] = None,
    ):
        """
        Given structure specifications, builds and factorizes a stiffness matrix.

        Args:
            lengths : (n,) lengths of each beam segment
            EIs: (n,) EIs of each beam segment
            fixed_rotation: (n+1,) boolean tensor whether the rotation of each node is fixed.
            fixed_translation: (n+1,) whether the translation of each node is fixed.
            thicknesses : (n,) thicknesses of each beam segment which is useful for computing max strain
        """
        assert (
            len(lengths)
            == len(EIs)
            == len(fixed_rotation) - 1
            == len(fixed_translation) - 1
        )
        assert (
            lengths.ndim
            == EIs.ndim
            == fixed_rotation.ndim
            == fixed_translation.ndim
            == 1
        )
        assert torch.all(EIs >= 0)
        assert fixed_rotation.dtype == fixed_translation.dtype == torch.bool
        self.n_beams = len(lengths)  # number of beams
        self.N = self.n_dof * self.n_nodes  # total number of DOFs
        assert thicknesses is None or self.n_beams == len(thicknesses)
        self.lengths = torch.asarray(lengths)
        self.EIs = torch.asarray(EIs)
        assert torch.all(self.EIs >= 0), str(self.EIs)
        self.fixed_rotation = fixed_rotation
        self.fixed_translation = fixed_translation
        self.thicknesses = thicknesses

        # get a boolean array of free DOFs
        fixed_dofs = self.dof_arr2vec(
            torch.hstack(
                [
                    torch.reshape(self.fixed_rotation, (-1, 1)),
                    torch.reshape(self.fixed_translation, (-1, 1)),
                ]
            )
        )
        self.free_dofs = torch.logical_not(fixed_dofs)

        # build the stiffness matrix
        Kglobal_lower_band = self.global_stiffness_matrix()

        # expand the lower band of the stiffness matrix to a full matrix
        # Note: this does not exploit the tridiagonal sparsity structure of the
        # stiffness matrix due to the lack of something like `scipy.linalg.solveh_banded`
        # in pytorch.
        # first fill the off-diagonal values
        Kglobal = (
            torch.diag(Kglobal_lower_band[1, :-1], diagonal=1)
            + torch.diag(Kglobal_lower_band[2, :-2], diagonal=2)
            + torch.diag(Kglobal_lower_band[3, :-3], diagonal=3)
        )
        # then make symmetric and add the diagonal value
        Kglobal = Kglobal + Kglobal.T + torch.diag(Kglobal_lower_band[0, :], diagonal=0)

        # apply the boundary conditions
        Kglobal = Kglobal[self.free_dofs, :][:, self.free_dofs]

        # factorize the stiffness matrix
        self.Kglobal_chol = self.factorize_stiffness_matrix(Kglobal)

    def get_displacements(
        self, uniform_loads: Optional[Tensor] = None, f: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the displacements of the beam.

        Args:
            uniform_loads : shape `(n,)` uniform loads on each beam segment. 
                Units should be in N/m.
            f : shape `(n+1, 2)` force vector to use at the nodes. Note that 
                other forces will be added to this.
        """
        if f is not None:
            assert f.size() == (self.n_nodes, self.n_dof)
        else:
            f = torch.zeros((self.n_nodes, self.n_dof))

        # add the uniform loads to the force vector
        if uniform_loads is not None:
            assert uniform_loads.size() == torch.Size([self.n_beams])
            assert uniform_loads.ndim == 1
            uniform_force = torch.asarray(uniform_loads) * self.lengths / 2
            uniform_moment = uniform_force * self.lengths / 6.0
            # apply equal force to the end of each beam segment
            f[:-1, 0] = f[:-1, 0] + uniform_force
            f[1:, 0] = f[1:, 0] + uniform_force
            # apply the appropriate moment to the end of each beam segment
            f[:-1, 1] = f[:-1, 1] + uniform_moment
            f[1:, 1] = f[1:, 1] - uniform_moment

        # apply the boundary conditions to f
        f_bc = self.dof_arr2vec(f)[self.free_dofs]

        # compute the displacements
        displacements_bc = torch.cholesky_solve(
            f_bc.reshape((-1, 1)), self.Kglobal_chol, upper=False,
        ).squeeze(dim=1)

        # fill-in the fixed coordinates
        displacements = torch.zeros(self.N)
        displacements[self.free_dofs] = displacements_bc
        displacements = self.dof_vec2arr(displacements)  # convert back to array form
        return displacements

    def get_max_strain(self, displacements):
        assert displacements.shape == (self.n_beams + 1, 2)
        assert self.thicknesses is not None

        # calculate curvature = d theta / ds
        theta = displacements[:, 1]
        dtheta = torch.diff(theta)
        curvature = dtheta / self.lengths

        # max axial strain
        maxStrain = -1 * curvature * self.thicknesses / 2
        return maxStrain

    def global_stiffness_matrix(self) -> Tensor:
        """
        Assembles the global stiffness matrix as a lower band as used by
        `scipy.linalg.solveh_banded`, for instance.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
        """
        Kglobal_lower_band = torch.zeros((4, self.N))
        for i, (l, EI) in enumerate(zip(self.lengths, self.EIs)):
            l2 = l ** 2
            Kglobal_lower_band[:, (2 * i) : (2 * (i + 2))] = (
                Kglobal_lower_band[:, (2 * i) : (2 * (i + 2))]
                + EI
                * torch.Tensor(
                    [
                        [12.0, 4.0 * l2, 12.0, 4.0 * l2],
                        [6.0 * l, -6.0 * l, -6.0 * l, 0.0],
                        [-12.0, 2.0 * l2, 0.0, 0.0],
                        [6.0 * l, 0.0, 0.0, 0.0],
                    ]
                )
                / l ** 3
            )
        return Kglobal_lower_band

    @property
    def n_dof(self) -> int:
        """ number of degrees of freedom per node """
        return 2

    @property
    def n_nodes(self) -> int:
        """ Number of degrees of freedom per node. """
        return self.n_beams + 1
