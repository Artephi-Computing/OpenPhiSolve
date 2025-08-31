from os.path import isfile, join
import numpy as np
import scipy as sp
from phisolve.problems.qubo import QUBO
from phisolve.problems.boxqp import BoxQP
from phisolve.problems.lcqp import LCQP
from phisolve.problems.miqp import MIQP
from typing import Union

def open_QP_file(file_name):
    with open(f"{file_name}.npy", "rb") as f:
        Q = np.load(f)
        b = np.load(f)
    return Q, b


def save_qubo(filename, Q, w, sol, objective_sense="min"):
    """Save QUBO problem data to a file using ``.npz`` format.

    Parameters
    ----------
    filename: str
    Q: array or csc_matrix
    w: array
    sol: array or str
    objective_sense: str ('min' or 'max')
    """
    if type(Q) == np.ndarray:
        np.savez(
            filename,
            problem_type="qubo",
            objective_sense=objective_sense,
            Q_type="dense",
            Q=Q,
            w=w,
            sol=sol,
        )

    elif type(Q) == sp.sparse._csc.csc_matrix:
        np.savez(
            filename,
            problem_type="qubo",
            objective_sense=objective_sense,
            Q_type="sparse",
            Q_data=Q.data,
            Q_indices=Q.indices,
            Q_indptr=Q.indptr,
            Q_shape=Q.shape,
            w=w,
            sol=sol,
        )

    else:
        raise TypeError("Q matrix type not supported.")


def save_boxqp(filename, Q, w, l, u, sol, objective_sense="min"):
    """Save BoxQP problem data to a file using ``.npz`` format.

    Parameters
    ----------
    filename: str
    Q: array or csc_matrix
    w: array
    l: array
    u: array
    sol: array or str
    objective_sense: str ('min' or 'max')
    """
    if type(Q) == np.ndarray:
        np.savez(
            filename,
            problem_type="boxqp",
            objective_sense=objective_sense,
            Q_type="dense",
            Q=Q,
            w=w,
            l=l,
            u=u,
            sol=sol,
        )

    elif type(Q) == sp.sparse._csc.csc_matrix:
        np.savez(
            filename,
            problem_type="boxqp",
            objective_sense=objective_sense,
            Q_type="sparse",
            Q_data=Q.data,
            Q_indices=Q.indices,
            Q_indptr=Q.indptr,
            Q_shape=Q.shape,
            w=w,
            l=l,
            u=u,
            sol=sol,
        )
    else:
        raise TypeError("Q matrix type not supported.")



def save_lcqp(filename, Q, w, A, b, C, d, l, u, sol, objective_sense="min"):
    """Save LCQP problem data to a file using `.npz` format.

    The LCQP problem is given by:
    min f(x) = 0.5 * x @ Q @ x + w @ x
    subject to A @ x <= b, C @ x = d, l <= x <= u

    Parameters
    ----------
    filename: str
    Q: array or csc_matrix
    w: array
    A: array or csc_matrix
    b: array
    C: array or csc_matrix
    d: array
    l: array
    u: array
    sol: array or str
    objective_sense: str, ('min' or 'max')
    """
    if isinstance(Q, np.ndarray):
        np.savez(
            filename,
            problem_type="lcqp",
            objective_sense=objective_sense,
            Q_type="dense",
            Q=Q,
            w=w,
            A_type="dense",
            A=A,
            b=b,
            C_type="dense",
            C=C,
            d=d,
            l=l,
            u=u,
            sol=sol,
        )
    elif isinstance(Q, sp.sparse._csc.csc_matrix):
        np.savez(
            filename,
            problem_type="lcqp",
            objective_sense=objective_sense,
            Q_type="sparse",
            Q_data=Q.data,
            Q_indices=Q.indices,
            Q_indptr=Q.indptr,
            Q_shape=Q.shape,
            w=w,
            A_type="sparse",
            A_data=A.data,
            A_indices=A.indices,
            A_indptr=A.indptr,
            A_shape=A.shape,
            b=b,
            C_type="sparse",
            C_data=C.data,
            C_indices=C.indices,
            C_indptr=C.indptr,
            C_shape=C.shape,
            d=d,
            l=l,
            u=u,
            sol=sol,
        )
    else:
        raise TypeError("Q matrix type not supported.")


def save_miqp(
    filename, Q, w, A, b, C, d, l, u, n_binary_vars, sol, objective_sense="min"
):
    """Save MIQP problem data to a file using ``.npz`` format.

    The MIQP problem is given by:
    minimize f(x) = 0.5 * x.T @ Q @ x + w.T @ x
    subject to: Ax <= b,
                l_i <= x_i <= u_i.
                for all i in I, x_i is in {l_i, u_i}

    Parameters
    ----------
    Q : np.ndarray
    w : np.ndarray
    A : np.ndarray
    b : np.ndarray
    C : np.ndarray
    d : np.ndarray
    bounds: Tuple
    n_binary_vars : int
    """

    if type(Q) == np.ndarray:
        np.savez(
            filename,
            problem_type="miqp",
            objective_sense=objective_sense,
            Q_type="dense",
            Q=Q,
            w=w,
            A_type="dense",
            A=A,
            b=b,
            C_type="dense",
            C=C,
            d=d,
            l=l,
            u=u,
            n_binary_vars=n_binary_vars,
            sol=sol,
        )

    elif type(Q) == sp.sparse._csc.csc_matrix:
        np.savez(
            filename,
            problem_type="miqp",
            objective_sense=objective_sense,
            Q_type="sparse",
            Q_data=Q.data,
            Q_indices=Q.indices,
            Q_indptr=Q.indptr,
            Q_shape=Q.shape,
            w=w,
            A_type="sparse",
            A_data=A.data,
            A_indices=A.indices,
            A_indptr=A.indptr,
            A_shape=A.shape,
            b=b,
            C_type="sparse",
            C_data=C.data,
            C_indices=C.indices,
            C_indptr=C.indptr,
            C_shape=C.shape,
            d=d,
            l=l,
            u=u,
            n_binary_vars=n_binary_vars,
            sol=sol,
        )
    else:
        raise TypeError("Q matrix type not supported.")

def read_2d_mat_from_buffer(buffer, s):
    if s+"_type" not in buffer:
        return buffer[s]
    elif buffer[s+"_type"] == "dense":
        return buffer[s]
    elif buffer[s+"_type"] == "sparse":
        cls = getattr(sp.sparse, "csc_matrix")
        M = cls(
            (buffer[s+"_data"], buffer[s+"_indices"], buffer[s+"_indptr"]),
            shape=buffer[s+"_shape"],
        )
        return M
    else:
        ValueError(s + "matrix type not supported.")

def load_instance(filename):
    """Load a problem instance from a file using ``.npz`` format.

    Parameters
    ----------
    filename: str
    """
    with np.load(filename, allow_pickle=True) as buffer:
        sign = 1 if buffer["objective_sense"] == "min" else -1
        sol = buffer["sol"] if buffer["sol"].dtype == "float" else None

        if buffer["problem_type"] == "qubo":
            Q = read_2d_mat_from_buffer(buffer, "Q")
            if buffer["Q_type"] == "dense":
                w = buffer["w"]
            elif buffer["Q_type"] == "sparse":
                w = buffer["w"]
                w = sp.sparse.diags(w)
            else:
                ValueError("Hessian type not supported.")
            return QUBO(sign * (Q + 2 * w)), sol

        elif buffer["problem_type"] == "boxqp":
            Q = read_2d_mat_from_buffer(buffer, "Q")
            w = buffer["w"]
            l = buffer["l"]
            u = buffer["u"]
            return (
                BoxQP(
                    sign * Q,
                    sign * w,
                    (l, u),
                ),
                sol,
            )

        elif buffer["problem_type"] == "lcqp":
            Q = read_2d_mat_from_buffer(buffer, "Q")
            w = buffer["w"]
            A = read_2d_mat_from_buffer(buffer, "A")
            b = buffer["b"]
            C = read_2d_mat_from_buffer(buffer, "C")
            d = buffer["d"]
            return (
                LCQP(
                    Q=sign * Q,
                    w=sign * w,
                    A=A,
                    b=b,
                    C=C,
                    d=d,
                    bounds=(l, u),
                ),
                sol,
            )

        elif buffer["problem_type"] == "miqp":
            Q = read_2d_mat_from_buffer(buffer, "Q")
            w = buffer["w"]
            A = read_2d_mat_from_buffer(buffer, "A")
            b = buffer["b"]
            C = read_2d_mat_from_buffer(buffer, "C")
            d = buffer["d"]
            n_binary_vars=buffer["n_binary_vars"],
            return (
                MIQP(
                    Q=sign * Q,
                    w=sign * w,
                    A=A,
                    b=b,
                    C=C,
                    d=d,
                    bounds=(l, u),
                    n_binary_vars=n_binary_vars
                ),
                sol,
            )