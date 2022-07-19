import torch
import torchmdo
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_less,
)
from .synthetic_functions import Rosenbrock
from pdb import set_trace

Tensor = torch.Tensor


class RosenbrockModel(torchmdo.Model):
    def __init__(self, x: Tensor):
        self.x = x
        self.objective_function = Rosenbrock(dim=self.x.numel())

    def compute(self):
        self.objective_value = self.objective_function(self.x)


class TestOptimizer:
    def setup_rosenbrock(self, constrained: bool):
        """
        Setup test rosenbrock optimization problem. If constrained then will set a
        upper bound constraint on the output that is always inactive, and a lower bound
        constraint on the output that is only active at the optimal solution.
        """
        model = RosenbrockModel(x=torch.zeros(2))
        design_variables = [
            torchmdo.DesignVariable(name="x"),
        ]
        objective = torchmdo.Minimize(
            name="objective_value",
            lower=torch.as_tensor(Rosenbrock.optimal_value) if constrained else None,
            upper=1e3 * torch.ones(()) if constrained else None,
        )
        constraints = []
        optimizer = torchmdo.Optimizer(
            initial_design_variables=design_variables,
            objective=objective,
            constraints=constraints,
            model=model,
        )
        return model, design_variables, objective, constraints, optimizer

    def test_optimize(self):
        """minimize the rosenbrock test function with and without an output constraint."""
        for constrained in [False, True]:
            (
                model,
                design_variables,
                objective,
                constraints,
                optimizer,
            ) = self.setup_rosenbrock(constrained=constrained)
            res = optimizer.optimize(maxiter=10000)
            assert_array_almost_equal(
                optimizer.variables_tensor.detach(),
                model.objective_function.minimizer.detach(),
                decimal=2.5 if constrained else 6,
                err_msg="Failed for constrained = %d" % int(constrained),
            )

    def test_gradients(self):
        """ 
        verifies that the objective gradient and constraint jacobian are correct
        at the initial guess.
        """
        (
            model,
            design_variables,
            objective,
            constraints,
            optimizer,
        ) = self.setup_rosenbrock(constrained=True)
        # check gradients
        objective_grad_error, constraints_jac_error = optimizer.check_grad()
        assert_array_less(objective_grad_error, 1e-5, "objective gradient failed.")
        assert_array_less(constraints_jac_error, 1e-5, "constraints jacobian failed.")


if __name__ == "__main__":
    TestOptimizer().test_optimize()
    # TestOptimizer().test_gradients()
