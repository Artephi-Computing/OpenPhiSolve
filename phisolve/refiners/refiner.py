import abc
from abc import ABC

from attr import define, field

from phisolve.problems import Problem

@define
class Refiner(ABC):
    iterations: int = field(default=1000)
    device: str = field(default="cpu")
    problem: Problem = field(default=None)

    @abc.abstractmethod
    def refine(self, samples, problem):
        pass