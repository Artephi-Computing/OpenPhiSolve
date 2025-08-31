from typing import Callable

import jax
from attr import field, dataclass
from jax import jit


# @dataclass
# class BackendParams:
#     n_shots: int = field(default=100)
#     n_steps: int = field(default=10000)
#     ballistic: bool = field(default=False)
#     verbose: bool = field(default=False)
#     device: str = field(default="cpu")
#     seed: int = field(default=None)
#     dt: float = field(default=0.2)
#     a0: float = field(default=1.0)
#     symplectic_integration: bool = field(default=False)
#     slow_a: bool = field(default=True)
#     lc_pr: float = field(default=3)   # Penalty ratio of linear constraint
#     constant_cons: bool = field(default=False)

#     def asdict(self):
#         return dict(self.__dict__)

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

        # if device == "cpu":
        #     new_dynamic = jit(new_dynamic, backend="cpu")
        # else:
        #     new_dynamic = jit(new_dynamic, backend="gpu")
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
