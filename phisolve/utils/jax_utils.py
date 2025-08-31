from jax import numpy as jnp
import numpy as np
from jax.experimental import sparse as jsp
import scipy as sp
from functools import partial

def jax_device(device):
    if device == "gpu":
        return "cuda"
    return device

def mat_norm(mat, ord=2):
    if mat.shape[0] == 0:
        return 0.
    if isinstance(mat, sp.sparse.spmatrix):
        return sp.sparse.linalg.norm(mat, ord)
    elif isinstance(mat, np.ndarray):
        return np.linalg.norm(mat.reshape(-1), ord=ord)
    return jnp.linalg.norm(mat.reshape(-1), ord=ord)

def inf_norm(mat):
    if isinstance(mat, np.ndarray):
        return mat_norm(mat.reshape(-1), ord=np.inf)
    return mat_norm(mat.reshape(-1), ord=jnp.inf)

def mat_square(mat):
    if isinstance(mat, sp.sparse.spmatrix):
        return mat.multiply(mat)
    else:
        return mat ** 2

def sp_sparse_mat_to_jsp_BCOO(sp_mat):
    sp_mat_coo = sp_mat.tocoo()
    data = jnp.array(sp_mat_coo.data)
    indices_np = np.stack((sp_mat_coo.row, sp_mat_coo.col)).T
    indices = jnp.array(indices_np)
    shape = sp_mat_coo.shape
    return jsp.BCOO((data, indices), shape=shape)

def sp_sparse_mat_to_jsp_BCSR(sp_mat):
    return jsp.BCSR.from_scipy_sparse(sp_mat)

# A matrix-vector multiplication compatible with sparse M in JAX's sparse format
def spcomp_matvec(M, v):
    def matvec(M, v):
        return M @ v
    if isinstance(M, sp.sparse.spmatrix):
        return jsp.sparsify(matvec)(M, v)
    else:
        return M @ v
    
def to_dense_mat(M):
    if isinstance(M, sp.sparse.spmatrix):
        return M.toarray()
    return M