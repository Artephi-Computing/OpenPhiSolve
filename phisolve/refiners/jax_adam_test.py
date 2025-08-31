import numpy as np
import pytest

from phisolve.problems.miqp import MIQP
from phisolve.refiners.jax_adam import JaxAdam

Q1 = np.array([[1.3, 1], [1, 1]], np.float64)
b1 = np.array([-1, -1], np.float64)

Q2 = np.array([[2, 0], [0, 2]], np.float64)
b2 = np.array([-1.2, -0.6], np.float64)


def jaxadam_helper(Q, b, expected):
    problem = MIQP(Q, b)
    samples = np.array([[1, 1], [0, 0]], np.float64)
    jaxadam = JaxAdam()
    pp_samples = jaxadam.refine(samples, problem)
    all_close_to_zero = np.all(np.isclose(pp_samples - expected, 0, atol=1e-3))
    assert all_close_to_zero


def test_jaxadam_correct_min():
    jaxadam_helper(Q1, b1, np.array([[0, 1], [0, 1]]))
    jaxadam_helper(Q2, b2, np.array([[0.6, 0.3], [0.6, 0.3]]))