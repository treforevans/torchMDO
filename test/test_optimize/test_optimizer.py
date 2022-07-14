import torch
import torchmdao
from torchmdao.examples.synthetic_functions import Rosenbrock
from numpy.testing import assert_almost_equal

Tensor = torch.Tensor


class RosenbrockModel(torchmdao.ComputeObject):
    def __init__(self, x: Tensor):
        self.x = x
        self.objective_function = Rosenbrock(dim=self.x.numel())

    def compute(self):
        self.objective_value = self.objective_function(self.x)


class TestOptimizer:
    def test_rosenbrock(self):
        """minimize the rosenbrock test function"""
        model = RosenbrockModel(x=torch.zeros(2))
        design_variables = [
            torchmdao.DesignVariable(name="x"),
        ]
        outputs = [torchmdao.Output(name="objective_value")]
        optimizer = torchmdao.Optimizer(
            initial_design_variables=design_variables,
            outputs=outputs,
            compute_object=model,
        )
        optimizer.optimize()
        assert_almost_equal(
            optimizer.variables_tensor.numpy(),
            model.objective_function.minimizer.numpy(),
            decimal=5,
        )


if __name__ == "__main__":
    TestOptimizer().test_rosenbrock()
