import numpy as np
import jax.numpy as jnp
import scipy as sp
from attr import field
from attrs import define
from typing import Tuple
from phisolve.problems.boxqp import BoxQP
from phisolve.utils.jax_utils import inf_norm, mat_norm, to_dense_mat

@define
class LCQP(BoxQP):
    """
    Class to define linearly constrained quadratic programs:
    minimize f(x) = 0.5 * x.T @ Q @ x + w.T @ x
    subject to: Ax <= b, Cx = d, lb <= x <= ub

    Parameters
    ----------
    Q : np.ndarray
    w : np.ndarray
    A : np.ndarray
    b : np.ndarray
    C : np.ndarray
    d : np.ndarray
    bounds : Tuple[np.ndarray, np.ndarray]
    """

    A: np.ndarray = field()
    b: np.ndarray = field()
    C: np.ndarray = field()
    d: np.ndarray = field()
    ncon_ineq: int = field()
    ncon_eq: int = field()

    @A.default
    def A_default(self):
        return np.zeros((0, self.w.shape[0]))

    @b.default
    def b_default(self):
        return np.array([])

    @C.default
    def C_default(self):
        return np.zeros((0, self.w.shape[0]))

    @d.default
    def d_default(self):
        return np.array([])

    @property
    def A_norm(self):
        return mat_norm(self.A)

    @property
    def b_norm(self):
        return mat_norm(self.b)

    @property
    def C_norm(self):
        return mat_norm(self.C)

    @property
    def d_norm(self):
        return mat_norm(self.d)

    @ncon_ineq.default
    def ncon_ineq_default(self):
        return self.b.shape[0]

    @ncon_eq.default
    def ncon_eq_default(self):
        return self.d.shape[0]

    @property
    def A2(self):
        return self.A.T @ self.A

    @property
    def C2(self):
        return self.C.T @ self.C

    @A.validator
    def a_validator(instance, field, val):
        if val.shape[0] != instance.b.shape[0]:
            raise ValueError("A and b dimensions don't match")

    @C.validator
    def c_validator(instance, field, val):
        if val.shape[0] != instance.d.shape[0]:
            raise ValueError("C and d dimensions don't match")

    def with_new_bounds(self, bounds):
        return LCQP(Q=self.Q, w=self.w, A=self.A, b=self.b, C=self.C, d=self.d, bounds=bounds)

    def calc_A_b_C_d_affine(self):
        k = (self.bounds[0] - self.bounds[1]) / 2
        t = (self.bounds[0] + self.bounds[1]) / 2

        if isinstance(self.A, sp.sparse.spmatrix):
            A_affine = self.A.multiply(k)
        else:
            A_affine = self.A * k
        b_affine = self.b - self.A @ t

        if isinstance(self.C, sp.sparse.spmatrix):
            C_affine = self.C.multiply(k)
        else:
            C_affine = self.C * k
        d_affine = self.d - self.C @ t

        return A_affine, b_affine, C_affine, d_affine

    def violation(self, x):
        return (
            np.fmax(0, self.A @ x - self.b),
            self.C @ x - self.d,
            np.fmax(0, self.bounds[0] - x),
            np.fmax(0, x - self.bounds[1]),
        )
    
    def vios(self, x):
        return self.violation(x)
    
    def max_vios(self, x):
        vio_ineq, vio_eq, vio_lb, vio_ub = self.vios(x)
        vio = np.concatenate((vio_ineq, np.abs(vio_eq), vio_lb, vio_ub))
        return np.max(vio)

    def to_dict(self):
        return dict(
            Q=self.Q,
            w=self.w,
            bounds=self.bounds,
            A=self.A,
            b=self.b,
            C=self.C,
            d=self.d,
        )

    def with_new_bounds(self, bounds):
        kwargs = self.to_dict()
        kwargs['bounds'] = bounds
        return LCQP(**kwargs)

    def clone(self):
        kwargs = self.to_dict()
        return LCQP(**kwargs)
    
    def to_dense(self):
        kwargs = self.to_dict()
        kwargs["Q"] = to_dense_mat(kwargs["Q"])
        kwargs["A"] = to_dense_mat(kwargs["A"])
        kwargs["C"] = to_dense_mat(kwargs["C"])
        return LCQP(**kwargs)

    def with_new_w(self, w):
        kwargs = self.to_dict()
        kwargs['w'] = w
        return LCQP(**kwargs)

    @staticmethod
    def inf_norm(mat):
        if mat.shape[0] == 0:
            return 0
        # return jnp.max((jnp.max(mat), -jnp.min(mat)))
        return jnp.linalg.norm(mat, ord=jnp.inf)

    def penalty_ratio_helper(self):
        max_Q_w = np.max((inf_norm(self.Q), inf_norm(self.w)))
        max_A_b = np.max((inf_norm(self.A), inf_norm(self.b)))
        if max_A_b == 0:
            return 1
        return max_Q_w / max_A_b

    def to_BoxQP_with_penalty(self, p):
        """
        Turn an LCQP (
            min. 0.5 * x.T @ Q @ x + w.T @ x, subj. to A @ x = b, lb <= x <= ub
            )
        to a BoxQP (equivalently,
            min. 0.5 * x.T @ Q @ x + w.T @ x + 0.5 * p * ||A @ x - b||^2, subj. to lb <= x <= ub
            )
        with a penalty coefficeint p.
        """
        Q_prime = self.Q + p * self.A2
        w_prime = self.w - p * self.b @ self.A
        return BoxQP(Q_prime, w_prime, self.bounds)

    def to_BoxQP_with_penalty_ratio(self, pr):
        """
        Args:
        pr: Penalty ratio. p=pr * max(|Q|, |w|) / max(|A|, |b|)
        """
        p = pr * self.penalty_ratio_helper()
        return self.to_BoxQP_with_penalty(p)

    def relKKT(self, x, y, z, boundary_eps=1e-6, return_details=False):
        r_primal_numerator, r_primal_denominator = 0, 0
        Qx = self.Q @ x
        xTQx = x @ Qx
        r_dual_numerator_vec = Qx + self.w
        r_dual_denominator = jnp.max(
            jnp.array((inf_norm(Qx), inf_norm(self.w)))
        )
        r_gap_numerator = xTQx + self.w @ x
        r_gap_denominator = 0.5 * xTQx
        if self.ncon_ineq > 0:
            Ax = self.A @ x
            ATy = self.A.T @ y
            bTy = self.b @ y
            r_primal_numerator = jnp.max(
                jnp.array((r_primal_numerator, jnp.max(Ax - self.b)))
            )
            r_primal_denominator = jnp.max(
                jnp.array(
                    (r_primal_denominator, inf_norm(Ax), inf_norm(self.b))
                )
            )
            r_dual_numerator_vec += ATy
            r_dual_denominator = jnp.max(
                jnp.array((r_dual_denominator, inf_norm(ATy)))
            )
            r_gap_numerator += bTy
            r_gap_denominator += bTy
        if self.ncon_eq > 0:
            Cx = self.C @ x
            CTz = self.C.T @ z
            dTz = self.d @ z
            r_primal_numerator = jnp.max(
                jnp.array((r_primal_numerator, inf_norm(Cx - self.d)))
            )
            r_primal_denominator = jnp.max(
                jnp.array(
                    (r_primal_denominator, inf_norm(Cx), inf_norm(self.d))
                )
            )
            r_dual_numerator_vec += CTz
            r_dual_denominator = jnp.max(
                jnp.array((r_dual_denominator, inf_norm(CTz)))
            )
            r_gap_numerator += dTz
            r_gap_denominator += dTz
        r_gap_denominator = jnp.max(
            jnp.array((jnp.abs(self.obj(x)), jnp.abs(r_gap_denominator)))
        )
        r_primal = r_primal_numerator / (1 + r_primal_denominator)
        on_boundary = jnp.less(x, self.bounds[0] + boundary_eps) | jnp.less(
            self.bounds[1] - boundary_eps, x
        )
        r_dual_numerator_vec = ~on_boundary * r_dual_numerator_vec
        r_dual = inf_norm(r_dual_numerator_vec) / (1 + r_dual_denominator)
        r_gap = jnp.abs(r_gap_numerator) / (1 + r_gap_denominator)
        if return_details:
            # jax.debug.print("{x}, {y}, {z}", x=x, y=y, z=z)
            # jax.debug.print("Residuals: {tmp}", tmp=jnp.array((r_primal, r_dual, r_gap)))
            return (r_primal, r_dual, r_gap)
        return jnp.max(jnp.array((r_primal, r_dual, r_gap)))

    def obj_batch(self, X):
        """Delegate to BoxQP batched objective (supports NumPy and JAX arrays)."""
        # BoxQP implements obj_batch and obj_batch_jax_vmap
        from phisolve.problems.boxqp import BoxQP
        return BoxQP.obj_batch(self, X)

    def max_vios_batch(self, X):
        """Vectorized max-violation per sample for LCQP.

        Accepts X shape (n_samples, nvar). Returns per-sample max violation.
        Supports both NumPy and JAX arrays.
        """
        # JAX path
        if isinstance(X, jnp.ndarray):
            # prepare jax arrays if possible
            if hasattr(self, 'prepare_jax'):
                try:
                    self.prepare_jax()
                except Exception:
                    pass

            # ensure jax attributes exist
            if not hasattr(self, 'A_j'):
                self.A_j = jnp.asarray(self.A)
            if not hasattr(self, 'b_j'):
                self.b_j = jnp.asarray(self.b)
            if not hasattr(self, 'C_j'):
                self.C_j = jnp.asarray(self.C)
            if not hasattr(self, 'd_j'):
                self.d_j = jnp.asarray(self.d)
            if not hasattr(self, 'bounds_j'):
                self.bounds_j = (jnp.asarray(self.bounds[0]), jnp.asarray(self.bounds[1]))

            X = jnp.atleast_2d(X)
            X_t = X.T
            n_samples = X.shape[0]
            vio_ineq = jnp.maximum(0.0, self.A_j @ X_t - self.b_j[:, None]) if self.ncon_ineq > 0 else jnp.zeros((0, n_samples))
            vio_eq = self.C_j @ X_t - self.d_j[:, None] if self.ncon_eq > 0 else jnp.zeros((0, n_samples))
            vio_lb = jnp.maximum(0.0, self.bounds_j[0][:, None] - X_t)
            vio_ub = jnp.maximum(0.0, X_t - self.bounds_j[1][:, None])
            vio = jnp.concatenate([vio_ineq, jnp.abs(vio_eq), vio_lb, vio_ub], axis=0)
            return jnp.max(vio, axis=0)

        # NumPy path
        X = np.atleast_2d(np.asarray(X))
        X_t = X.T
        n_samples = X.shape[0]
        vio_ineq = np.maximum(0.0, self.A @ X_t - self.b[:, None]) if self.ncon_ineq > 0 else np.zeros((0, n_samples))
        vio_eq = self.C @ X_t - self.d[:, None] if self.ncon_eq > 0 else np.zeros((0, n_samples))
        vio_lb = np.maximum(0.0, self.bounds[0][:, None] - X_t)
        vio_ub = np.maximum(0.0, X_t - self.bounds[1][:, None])
        vio = np.concatenate([vio_ineq, np.abs(vio_eq), vio_lb, vio_ub], axis=0)
        return np.max(vio, axis=0)
