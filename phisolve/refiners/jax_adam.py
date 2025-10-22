import optax
from attr import define, field
from jax import jit, vmap, value_and_grad, device_put, devices
import jax.numpy as jnp
from phisolve.refiners.refiner import Refiner
import scipy as sp
import jax.experimental.sparse as jsp
from phisolve.utils.jax_utils import sp_sparse_mat_to_jsp_BCSR, spcomp_matvec
from functools import partial
import jax


@define
class JaxAdam(Refiner):
    learning_rate: float = field(default=1e-1)
    iterations: int = field(default=300)
    device: str = field(default="cpu")
    vmappable: bool = field(default=True)

    def run_jax(self, samples):
        optimizer = optax.adam(self.learning_rate)

        samples = device_put(samples, devices(self.device)[0])
        opt_state = optimizer.init(samples)

        Q, w = self.problem.Q, self.problem.w

        def grad_gen(Q, x):
            return spcomp_matvec(Q, x) + w
        if isinstance(Q, sp.sparse.spmatrix):
            Q = sp_sparse_mat_to_jsp_BCSR(Q)
        grad = partial(grad_gen, Q)
        
        def step(samples, opt_state):
            # loss_values, grads = vmap(value_and_grad(self.problem.obj))(samples)
            grads = vmap(grad)(samples)
            updates, opt_state = optimizer.update(grads, opt_state, samples)
            samples = vmap(optax.apply_updates)(samples, updates)
            samples = jnp.minimum(jnp.maximum(samples, self.problem.bounds[0]), self.problem.bounds[1])
            return samples, opt_state

        for _ in range(self.iterations):
            samples, opt_state = jit(step)(samples, opt_state)
        return samples

    def refine(self, samples, problem):
        self.problem = problem
        samples = jnp.array(samples, jnp.float32)
        processed_samples = self.run_jax(samples)
        return processed_samples
