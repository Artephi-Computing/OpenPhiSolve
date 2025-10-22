import logging
from functools import partial
from typing import Union
import os
from jax.sharding import NamedSharding, Mesh
from jax.experimental import mesh_utils
from jax.nn import relu
import jax
from attr import define, field, asdict
import numpy as np
import jax.numpy as jnp
import scipy as sp
from jax import vmap, grad
from phisolve.utils.jax_utils import sp_sparse_mat_to_jsp_BCOO, sp_sparse_mat_to_jsp_BCSR, mat_square, mat_norm, spcomp_matvec

from phisolve.backends.backend import Backend
from phisolve.utils.integrator import Integrator
from phisolve.problems.problem import Problem

_logger = logging.getLogger(__name__)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@define
class QIHD(Backend):
    """
    Backend instantiation for QIHD.
    """
    n_shots: int = field(default=100)
    n_steps: int = field(default=10000)
    ballistic: bool = field(default=False)
    verbose: bool = field(default=False)
    device: str = field(default="cpu")
    seed: int = field(default=None)
    dt: float = field(default=0.2)
    a0: float = field(default=1.0)
    symplectic_integration: bool = field(default=True)
    slow_a: bool = field(default=True)
    lc_pr: float = field(default=3)   # Penalty ratio of linear constraint
    constant_cons: bool = field(default=False)
    c0_multiplier: float = field(default=0.5)
    
    def c0_default(self, Q, w, A, b, C, d):
        n_dim = w.shape[0]
        num2_sum = mat_square(Q).sum() + (w ** 2).sum() + \
            mat_square(A).sum() + np.sum(b ** 2) +\
            mat_square(C).sum() + np.sum(d ** 2)
        num_cnt = (n_dim + A.shape[0] + C.shape[0]) * (n_dim + 1)
        C = np.sqrt(num2_sum / num_cnt)
        return self.c0_multiplier / (C*np.sqrt(n_dim))

    def round_to_nearest_device_count(self, n_shots, device_count):
        if device_count == 1:
            return n_shots
        return n_shots - (n_shots % device_count)

    def generate_samples(
        self,
        # params: BackendParams,
        problem: Problem,
    ):
        _logger.debug(
            "Generating samples with %s based on %s",
            self.__class__.__name__,
            asdict(self),
        )
        self.problem = problem # update problem for debugging purpose
        n_devices = 1 if self.device == "cpu" else jax.device_count('gpu')
        n_shots = self.round_to_nearest_device_count(self.n_shots, n_devices)
        n_steps = self.n_steps
        ballistic = self.ballistic
        a0 = self.a0
        dt = self.dt
        si = self.symplectic_integration
        slow_a = self.slow_a
        constant_cons = self.constant_cons

        Q, w, A, b, C, d = problem.Q, problem.w, problem.A, problem.b, problem.C, problem.d
        n_dim = w.shape[0]
        n_binary_vars = problem.n_binary_vars
        c0 = self.c0_default(Q, w, A, b, C, d)

        norm = partial(mat_norm, ord=np.inf)
        Q_norm, w_norm = norm(Q), norm(w)
        A_norm, b_norm, A_pen = norm(A), norm(b), 1.
        if A_norm > 0 or b_norm > 0:
            A_pen = self.lc_pr * np.max((Q_norm, w_norm)) / np.max((A_norm, b_norm))
        C_norm, d_norm, C_pen = norm(C), norm(d), 1.
        if C_norm > 0 or d_norm > 0:
            C_pen = self.lc_pr * np.max((Q_norm, w_norm)) / np.max((C_norm, d_norm))

        if isinstance(Q, sp.sparse.spmatrix):
            Q = sp_sparse_mat_to_jsp_BCSR(Q)
        if isinstance(A, sp.sparse.spmatrix):
            A = sp_sparse_mat_to_jsp_BCSR(A)
        if isinstance(C, sp.sparse.spmatrix):
            C = sp_sparse_mat_to_jsp_BCSR(C)

        I = jnp.zeros(n_dim)
        I = I.at[:n_binary_vars].set(1)
        J = 1 - I

        def x_tilde(x):
            return I * jnp.sign(x) + J * x
        
        def sb_step(carry, a):
            def sb_step_one_sample(x, y):
                def V_binary_pen(x):
                    binary_x = I * x
                    return 0.5 * (a0 - a) * jnp.sum(binary_x**2)

                def V_ineq_con(x):
                    Ax_minus_b = spcomp_matvec(A, x) - b
                    if constant_cons:
                        return A_pen * jnp.sum(relu(Ax_minus_b))
                    return a * A_pen * jnp.sum(relu(Ax_minus_b))

                def V_eq_con(x):
                    Cx_minus_d = spcomp_matvec(C, x) - d
                    if constant_cons:
                        return C_pen * jnp.sum(jnp.abs(Cx_minus_d))
                    return a * C_pen * jnp.sum(jnp.abs(Cx_minus_d))

                def V_f(x):
                    fx = 0.5 * x @ spcomp_matvec(Q, x) + w @ x
                    return c0 * fx

                @jax.custom_gradient
                def V_f_disc(x):
                    fx = 0.5 * x @ spcomp_matvec(Q, x) + w @ x
                    return c0 * fx, lambda g: g * c0 * (spcomp_matvec(Q, x_tilde(x)) + w)

                y -= dt * grad(V_binary_pen)(x)
                if not si:
                    x += dt * a0 * y

                if ballistic:
                    y -= dt * (grad(V_f)(x) + grad(V_ineq_con)(x) + grad(V_eq_con)(x))
                else:
                    y -= dt * (grad(V_f_disc)(x) + grad(V_ineq_con)(x_tilde(x)) + grad(V_eq_con)(x_tilde(x)))

                if si:
                    x += dt * a0 * y
                y = jnp.where(jnp.abs(x) > 1.0, 0.0, y)
                x = jnp.clip(x, -1.0, 1.0)
                return x, y

            xs, ys = carry
            return vmap(vmap(sb_step_one_sample))(xs, ys)

        if slow_a:
            a_schedule = np.arange(n_steps, dtype=np.float32) / (n_steps - 1000)
        else:
            a_schedule = np.arange(n_steps, dtype=np.float32) / 1000
        a_schedule = np.where(a_schedule > a0, a0, a_schedule)
        a_schedule = jnp.array(a_schedule)
        if self.seed is not None:
            rng = np.random.default_rng(seed=self.seed)
        else:
            rng = np.random.default_rng()
        array = 2 * rng.random((2, n_devices, int(n_shots / n_devices), n_dim)) - 1
        x = jnp.array(array[0])
        y = jnp.array(array[1])
        if self.device == "gpu":
            mesh = Mesh(mesh_utils.create_device_mesh((n_devices,)), ('samples'))
            sharding = NamedSharding(mesh, jax.sharding.PartitionSpec('samples', None, None))
            x = jax.device_put(x, sharding)
            y = jax.device_put(y, sharding)
        result = Integrator.integrate(sb_step, (x, y), a_schedule, device=self.device)
        x_final = result[0]
        x_final = vmap(vmap(x_tilde))(x_final)
        x_final = jax.device_get(x_final)
        x_final = x_final.reshape(n_shots, n_dim)
        return x_final
