from functools import partial

from attr import define, field
from jax import jit, grad, jacfwd, jacrev
import jax.numpy as jnp
from scipy.optimize import Bounds
from cyipopt import minimize_ipopt
from phisolve.refiners.refiner import Refiner
from phisolve.problems.boxqp import BoxQP
from phisolve.problems.lcqp import LCQP


@define
class IpoptMinimize(Refiner):
    tol: float = field(default=1e-6)
    max_iter: float = field(default=100)
    device: str = field(default="cpu")

    def refine(self, samples, problem):
        self.problem = problem.to_dense()
        refined_samples = []
        backend = "gpu" if self.device == "cuda" else self.device
        eval_obj = jit(self.problem.obj, backend=backend)
        eval_grad = jit(grad(eval_obj), backend=backend)
        eval_hess = jit(jacrev(jacfwd(eval_obj)), backend=backend)
        
        lower_bound = jnp.array(self.problem.bounds[0])
        upper_bound = jnp.array(self.problem.bounds[1])
        bounds = Bounds(lower_bound, upper_bound)
        if isinstance(self.problem, BoxQP):
            constraints = ()
        elif isinstance(self.problem, LCQP):
            eval_cons = jit(self.problem.cons, backend=backend)
            eval_cons_grad = jit(jacfwd(eval_cons), backend=backend)
            eval_cons_hess = jit(jacrev(jacfwd(eval_cons)), backend=backend)

            constraints = dict(
                type='eq', 
                fun=eval_cons, 
                jac=eval_cons_grad, 
                hess=eval_cons_hess
            )

        for k in range(len(samples)):
            if samples[k] is None:
                refined_samples.append(None)
                continue
            result = minimize_ipopt(
                eval_obj,
                samples[k],
                jac=eval_grad,
                hess=eval_hess,
                bounds=bounds,
                constraints=constraints,
                options={"tol": self.tol, "max_iter": self.max_iter},
            )
            refined_samples.append(result.x)
        
        return refined_samples

