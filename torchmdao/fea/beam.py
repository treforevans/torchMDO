import numpy as np
from scipy.linalg import solveh_banded
from pdb import set_trace


class BaseFEA:
    def dof_vec2arr(self, vec):
        """
        converts a vector of size N=n_nodes*n_dof to an array of shape (n_nodes, n_dof) where
        each row corresponds to the degrees of freedom of a node.
        """
        arr = np.reshape(vec, (self.n_nodes, self.n_dof), order="C")
        return arr

    def dof_arr2vec(self, arr):
        """
        converts an array of shape (n_nodes, n_dof) (where each row corresponds to the degrees of freedom of a node)
        to a vector of size N=n_nodes*n_dof in the appropriate order.
        """
        vec = np.reshape(arr, (self.n_nodes * self.n_dof,), order="C")
        return vec


class BeamFEA(BaseFEA):
    def __init__(
        self, lengths, EIs, fixed_rotation, fixed_translation, thicknesses=None
    ):
        """
        Given structure specifications, builds and factorizes a stiffness matrix.

        Inputs:
            lengths : (n,) lengths of each beam segment
            EIs: (n,) EIs of each beam segment
            fixed_rotation: (n+1,) whether the rotation of each node is fixed.
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
            np.ndim(lengths)
            == np.ndim(EIs)
            == np.ndim(fixed_rotation)
            == np.ndim(fixed_translation)
            == 1
        )
        self.n_beams = len(lengths)  # number of beams
        self.n_nodes = self.n_beams + 1  # number of nodes
        self.n_dof = 2  # number of degrees of freedom per node
        self.N = self.n_dof * self.n_nodes  # total number of DOFs
        assert thicknesses is None or self.n_beams == len(thicknesses)
        self.lengths = np.asarray(lengths)
        self.EIs = np.asarray(EIs)
        assert np.all(self.EIs >= 0), str(self.EIs)
        self.fixed_rotation = fixed_rotation
        self.fixed_translation = fixed_translation
        self.thicknesses = thicknesses

        # get a boolean array of free DOFs
        fixed_dofs = self.dof_arr2vec(
            np.hstack(
                [
                    np.reshape(self.fixed_rotation, (-1, 1)),
                    np.reshape(self.fixed_translation, (-1, 1)),
                ]
            )
        )
        self.free_dofs = np.logical_not(fixed_dofs)

        # build the stiffness matrix
        self._assemble_stiffness()

        # apply the boundary conditions
        assert np.all(
            np.where(fixed_dofs)[0] == [0, 1]
        ), "currently only implemented for a clamped cantilevered beam"  # TODO: we can implement arbitrary boundary conditions by deleting the correct elements from the proceeding dofs in K_banded_full but it's not done yet
        self.K_lower_band = self.Kfull_lower_band[:, 2:]  # clamped beam

    def get_displacements(self, uniform_loads=None, f=None):
        """
        Compute the displacements of the beam.
            uniform_loads : (n,) uniform loads on each beam segment. Units should be in N/m.
            f : (n+1, 2) force vector to use. Note that other forces will be added to this.
        """
        if f is not None:
            assert np.shape(f) == (self.n_nodes, self.n_dof)
        else:
            f = np.zeros((self.n_nodes, self.n_dof))

        # add the uniform loads to the force vector
        if uniform_loads is not None:
            assert np.size(uniform_loads) == self.n_beams
            assert np.ndim(uniform_loads) == 1
            uniform_force = np.asarray(uniform_loads) * self.lengths / 2
            uniform_moment = uniform_force * self.lengths / 6.0
            # apply equal force to the end of each beam segment
            f[:-1, 0] += uniform_force
            f[1:, 0] += uniform_force
            # apply the appropriate moment to the end of each beam segment
            f[:-1, 1] += uniform_moment
            f[1:, 1] -= uniform_moment

        # apply the boundary conditions to f
        f_bc = self.dof_arr2vec(f)[self.free_dofs]

        # compute the displacements
        displacements_bc = solveh_banded(ab=self.K_lower_band, b=f_bc, lower=True,)

        # fill-in the fixed coordinates
        displacements = np.zeros(self.N)
        displacements[self.free_dofs] = displacements_bc
        displacements = self.dof_vec2arr(displacements)  # convert back to array form
        return displacements

    def get_max_strain(self, displacements):
        assert displacements.shape == (self.n_beams + 1, 2)
        assert self.thicknesses is not None

        # calculate curvature = d theta / ds
        theta = displacements[:, 1]
        dtheta = np.diff(theta)
        curvature = dtheta / self.lengths

        # max axial strain
        maxStrain = -1 * curvature * self.thicknesses / 2
        return maxStrain

    def _assemble_stiffness(self):
        self.Kfull_lower_band = np.zeros((4, self.N))
        for i, (l, EI) in enumerate(zip(self.lengths, self.EIs)):
            l2 = l ** 2
            self.Kfull_lower_band[:, (2 * i) : (2 * (i + 2))] += (
                EI
                * np.array(
                    [
                        [12.0, 4.0 * l2, 12.0, 4.0 * l2],
                        [6.0 * l, -6.0 * l, -6.0 * l, 0.0],
                        [-12.0, 2.0 * l2, 0.0, 0.0],
                        [6.0 * l, 0.0, 0.0, 0.0],
                    ]
                )
                / l ** 3
            )

