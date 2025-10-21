from typing import Callable

import jax
from attr import field, dataclass
from jax import jit


class Integrator:

    def integrate(
        dynamic: Callable,
        initial_state,
        time_dependent_variables=None,
        time_steps=None,
        device="cpu",
    ):
        def new_dynamic(state, variables):
            return dynamic(state, variables), None

        new_dynamic = jit(new_dynamic)
        if time_dependent_variables is None:
            assert time_steps is not None
            final_state, _ = jax.lax.scan(new_dynamic, initial_state, length=time_steps)
        else:
            assert time_steps is None or time_steps == time_dependent_variables.shape[0]
            final_state, _ = jax.lax.scan(
                new_dynamic, initial_state, time_dependent_variables
            )
        return final_state
