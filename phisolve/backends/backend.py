import abc
from abc import ABC

from attr import define, field

from phisolve.problems.problem import Problem

@define
class Backend(ABC):
    n_steps: int = field(default=100)
    n_shots: int = field(default=10000)
    device: str = field(default='cpu')
    dt: float = field(default=0.2)
    seed: int = field(default=None)
    problem: Problem = field(default=None)

    @abc.abstractmethod
    def generate_samples(self, problem):
        pass