from functools import partial

import numpy as np
from attr import define
from attrs import field
from jax import jit, grad
import jax.numpy as jnp
from phisolve.problems.miqp import MIQP
from scipy.optimize import Bounds, minimize, LinearConstraint
from phisolve.refiners.refiner import Refiner
from phisolve.problems.boxqp import BoxQP
from phisolve.problems.lcqp import LCQP


@define
class ScipyMinimize(Refiner):
    method: str = field(default="TNC")
    options: dict = field(default=dict(gtol=1e-6, eps=1e-9))
    device: str = field(default="cpu")

    def refine(self, samples, problem):
        samples = np.array(samples)
        self.problem = problem.to_dense()
        refined_samples = []
        backend = "gpu" if self.device == "cuda" else self.device
        eval_jit = jit(self.problem.obj, backend=backend)
        eval_grad = jit(grad(eval_jit), backend=backend)

        lower_bound = jnp.array(self.problem.bounds[0])
        upper_bound = jnp.array(self.problem.bounds[1])
        bounds = Bounds(lower_bound, upper_bound)
        if isinstance(self.problem, BoxQP) or (isinstance(self.problem, MIQP) and self.problem.ncon_eq_default == 0):
            constraints = ()
        elif isinstance(self.problem, LCQP):
            constraints = [LinearConstraint(self.problem.A, self.problem.b, self.problem.b)]

        for k in range(len(samples)):
            if samples[k] is None:
                refined_samples.append(None)
                continue
            result = minimize(
                eval_jit,
                samples[k].astype("float64"),
                method=self.method,
                jac=eval_grad,
                bounds=bounds,
                constraints=constraints,
                options=self.options,
            )
            refined_samples.append(result.x)
        return refined_samples

