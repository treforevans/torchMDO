import torch
from abc import ABC, abstractmethod
from gpytorch.utils.cholesky import psd_safe_cholesky
from pdb import set_trace

Tensor = torch.Tensor


class BaseFEA(ABC):
    def dof_vec2arr(self, vec) -> Tensor:
        """
        converts a vector of size N=n_nodes*n_dof to an array of shape (n_nodes, n_dof) where
        each row corresponds to the degrees of freedom of a node.
        """
        arr = torch.reshape(vec, (self.n_nodes, self.n_dof))
        return arr

    def dof_arr2vec(self, arr) -> Tensor:
        """
        converts an array of shape (n_nodes, n_dof) (where each row corresponds to the degrees of freedom of a node)
        to a vector of size N=n_nodes*n_dof in the appropriate order.
        """
        vec = torch.reshape(arr, (self.n_nodes * self.n_dof,))
        return vec

    def factorize_stiffness_matrix(self, K: Tensor) -> Tensor:
        """
        Factorizes the symmetric Positive definite stiffness matrix K using the
        lower triangular Cholesky factorization.
        """
        return psd_safe_cholesky(K, jitter=1e-7, max_tries=3, upper=False)

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """ Number of degrees of freedom per node. """
        pass

    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """ Number of degrees of freedom per node. """
        pass
