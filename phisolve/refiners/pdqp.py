from attr import define, field
from jax import jit, vmap, devices, lax
import jax
from phisolve.problems.lcqp import LCQP
from phisolve.problems.boxqp import BoxQP
import jax.numpy as jnp
from phisolve.refiners.refiner import Refiner
from phisolve.utils.jax_utils import jax_device
import mpax as mp

@define
class PDQP(Refiner):
    iterations: int = field(default=1000)
    max_K: int = field(default=1)
    device: str = field(default="cpu")
    vmappable: bool = field(default=True)
    
    def pdqp_main(self, samples, dual_vars = None):
        qp = mp.create_qp(self.problem.Q, 
                          self.problem.w, 
                          self.problem.C, 
                          self.problem.d, 
                          - self.problem.A, 
                          - self.problem.b,
                          self.problem.bounds[0],
                          self.problem.bounds[1])

        solver = mp.raPDHG(eps_abs=1e-4, 
                           eps_rel=1e-4, 
                           warm_start=True, 
                           iteration_limit=self.iterations*self.max_K,
                           display_frequency=1,
                           verbose=False)

        def solve_qp_with_dual(x, y, z):
            result = solver.optimize(qp, x, jnp.concatenate((z, y)))
            return result.primal_solution

        if dual_vars is None:
            dual_vars = (jnp.zeros((samples.shape[0], self.problem.A.shape[0])), 
                         jnp.zeros((samples.shape[0], self.problem.C.shape[0])))
        
        res = jax.vmap(solve_qp_with_dual)(samples, dual_vars[0], dual_vars[1])

        res = jax.device_get(res)
        return res

    def refine(self, samples, problem, dual_vars = None):
        if isinstance(problem, LCQP):
            self.problem = problem
        elif isinstance(problem, BoxQP):
            self.problem = LCQP(Q=problem.Q, w=problem.w, bounds=problem.bounds)
        else:
            raise ValueError("problem is not in {BoxQP, LCQP}.")

        samples = jnp.array(samples)
        processed_samples = jit(self.pdqp_main)(samples, dual_vars)
        return processed_samples