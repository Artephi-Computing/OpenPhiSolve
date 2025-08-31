import numpy as np
from attr import field
from attrs import define
from phisolve.problems.lcqp import LCQP
import scipy as sp
from phisolve.utils.jax_utils import to_dense_mat


@define
class MIQP(LCQP):
    """
    Class to define Mixed-Integer quadratic programs:
    minimize f(x) = 0.5 * x.T @ Q @ x + w.T @ x
    subject to: Ax <= b,
                Cx = d,
                l_i <= x_i <= u_i.
                for all i in I, x_i is in {l_i, u_i}

    Parameters
    ----------
    n_binary_vars
    """

    n_binary_vars: int = field()

    @n_binary_vars.default
    def n_binary_vars_default(self):
        return 0

    def to_dict(self):
        return dict(
            Q=self.Q,
            w=self.w,
            bounds=self.bounds,
            A=self.A,
            b=self.b,
            C=self.C,
            d=self.d,
            n_binary_vars=self.n_binary_vars,
        )

    def with_new_bounds(self, bounds):
        kwargs = self.to_dict()
        kwargs['bounds'] = bounds
        return MIQP(**kwargs)

    def clone(self):
        kwargs = self.to_dict()
        return MIQP(**kwargs)

    def to_dense(self):
        kwargs = self.to_dict()
        kwargs["Q"] = to_dense_mat(kwargs["Q"])
        kwargs["A"] = to_dense_mat(kwargs["A"])
        kwargs["C"] = to_dense_mat(kwargs["C"])
        return MIQP(**kwargs)

    def fix_binary_values(self, binary_values):
        Q_p = self.Q[self.n_binary_vars :, self.n_binary_vars :]
        w_p = self.w[self.n_binary_vars :] + \
            self.Q[self.n_binary_vars :, : self.n_binary_vars] @ binary_values
        A_p = self.A[:, self.n_binary_vars :]
        b_p = self.b - self.A[:, : self.n_binary_vars] @ binary_values
        C_p = self.C[:, self.n_binary_vars :]
        d_p = self.d - self.C[:, : self.n_binary_vars] @ binary_values
        l_p = self.bounds[0][self.n_binary_vars :]
        u_p = self.bounds[1][self.n_binary_vars :]
        return LCQP(Q=Q_p, w=w_p, A=A_p, b=b_p, C=C_p, d=d_p, bounds=(l_p, u_p))
