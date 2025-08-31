from typing import List, Tuple

import numpy as np
from attr import field
from attrs import define
from numpy import ndarray

from phisolve.problems.problem import Problem
from phisolve.problems.boxqp import BoxQP
from phisolve.utils.jax_utils import to_dense_mat


@define
class QUBO(Problem):
    """
    Class to define quadratic unconstrained binary optimization programs:
    minimize f(x) = 1/2 * x.T @ Q @ x
    subject to: x_i ∈ {0, 1}.
    (Note the coefficient 1/2 in f(x))

    Parameters
    ----------
    Q : np.ndarray
    nvar : int
    """

    Q: np.ndarray = field()
    nvar: int = field()

    @nvar.default
    def nvar_default(self):
        return self.Q.shape[0]

    @nvar.validator
    def nvar_validate(instance, field, val):
        if val != instance.Q.shape[0]:
            raise ValueError("dimensions don't match.")

    def to_dict(self):
        return dict(
            Q=self.Q,
            w=np.zeros(self.Q.shape[0]),
        )

    def clone(self):
        return QUBO(self.Q)
    
    def to_dense(self):
        kwargs = self.to_dict()
        kwargs["Q"] = to_dense_mat(kwargs["Q"])
        return QUBO(**kwargs)

    def obj(self, x):
        return 0.5 * x @ self.Q @ x

    def grad(self, x):
        return self.Q @ x

    def with_new_bounds(self, bounds):
        pass

    def bounds(self):
        pass

    def affine_trans(self, x):
        return x

    def relax_to_BoxQP(self):
        new_Q = self.Q
        w = np.zeros(self.nvar)
        return BoxQP(new_Q, w)

    def feasibility_test(self, x, weak=True):
        for i in range(self.nvar):
            if weak and 0 <= x[i] and x[i] <= 1:
                continue
            elif not weak and (np.isclose(x[i], 0) or np.isclose(x[i], 1)):
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
        z = np.clip(x - self.grad(x), 0, 1)
        return np.sqrt(np.sum((x - z) ** 2))
