from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np
from attr import field
from attrs import define
from numpy import ndarray
import scipy as sp

from phisolve.problems.problem import Problem
from phisolve.utils.jax_utils import to_dense_mat


@define
class BoxQP(Problem):
    """
    Class to define box-constrained quadratic programs:
    minimize f(x) = 0.5 * x.T @ Q @ x + w.T @ x
    subject to: l_i <= x_i <= u_i.

    Parameters
    ----------
    Q : Union[np.ndarray, sp.sparse.spmatrix]
    w : np.ndarray
    bounds: Tuple
    nvar : int
    """

    Q: Union[np.ndarray, sp.sparse.spmatrix] = field()
    w: np.ndarray = field()
    bounds: Tuple[ndarray, ndarray] = field()
    nvar: int = field()

    @bounds.default
    def bounds_default(self):
        return np.zeros(self.w.shape[0]), np.ones(self.w.shape[0])

    @nvar.default
    def nvar_default(self):
        return self.w.shape[0]

    @nvar.validator
    def nvar_validate(instance, field, val):
        if val != instance.Q.shape[0] or instance.Q.shape[0] != instance.w.shape[0]:
            raise ValueError("dimensions don't match.")

    @property
    def Q_norm(self):
        if isinstance(self.Q, np.ndarray):
            return np.linalg.norm(self.Q)
        return jnp.linalg.norm(self.Q)

    @property
    def w_norm(self):
        if isinstance(self.w, np.ndarray):
            return np.linalg.norm(self.w)
        return jnp.linalg.norm(self.w)

    def to_dict(self):
        return dict(
            Q=self.Q,
            w=self.w,
            bounds=self.bounds,
        )

    def with_new_bounds(self, bounds):
        kwargs = self.to_dict()
        kwargs['bounds'] = bounds
        return BoxQP(**kwargs)

    def clone(self):
        kwargs = self.to_dict()
        return BoxQP(**kwargs)
    
    def to_dense(self):
        kwargs = self.to_dict()
        kwargs["Q"] = to_dense_mat(kwargs["Q"])
        return BoxQP(**kwargs)

    def obj(self, x):
        return 0.5 * x @ self.Q @ x + self.w @ x

    def grad(self, x):
        return self.Q @ x + self.w

    def affine_trans(self, x):
        k = (self.bounds[0] - self.bounds[1]) / 2
        t = (self.bounds[0] + self.bounds[1]) / 2

        return k*x + t
    
    def vios(self, x):
        return (
            np.fmax(0, self.bounds[0] - x),
            np.fmax(0, x - self.bounds[1]),
        )
    
    def max_vios(self, x):
        vio_lb, vio_ub = self.vios(x)
        vio = np.concatenate((vio_lb, vio_ub))
        return np.max(vio)

    def calc_Q_w_affine(self):
        k = (self.bounds[0] - self.bounds[1]) / 2
        t = (self.bounds[0] + self.bounds[1]) / 2

        if isinstance(self.Q, sp.sparse.spmatrix):
            diag_k = sp.sparse.spdiags(k, 0, self.nvar, self.nvar)
            Q_affine = diag_k @ self.Q @ diag_k
            kQ = diag_k @ self.Q
            Qk = self.Q @ diag_k
            w_affine = 0.5 * (kQ + Qk) @ t + k * self.w
        else:
            Qk = self.Q * k
            kQ = (self.Q.T * k).T
            Q_affine = kQ * k
            w_affine = 0.5 * (kQ + Qk) @ t + k * self.w
        return Q_affine, w_affine

    def feasibility_test(self, x):
        for i in range(self.nvar):
            if self.bounds[0][i] <= x[i] and x[i] <= self.bounds[1][i]:
                continue
            else:
                return False
        return True

    def first_der_test(self, x, *args):
        """
        First derivative test (KKT conditions)
        violation of the KKT condition = ||x - P(x - nabla_x f(x), l, u)||_2
        When the violation is zero, KKT condition for boxQP is satisfied.
        Reference: Section 17.4, pp520, Nocedal & Wright
        """
        z = np.clip(x - self.grad(x), self.bounds[0], self.bounds[1])
        return np.sqrt(np.sum((x - z) ** 2))

    def decode(self, binary_values):
        binary_values = np.array(binary_values)
        return np.where(binary_values > 0.5, 1, 0)
