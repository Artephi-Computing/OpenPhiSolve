from typing import List

import jax
from attr import define, field
import numpy as np
from phisolve.problems import BoxQP


@define
class Response:
    problem_instance = field()
    coarse_samples: List[List[float]] = field()
    sample_counts: np.ndarray = field()
    refined_samples: List[List[float]] = field(default=None)
    time = field(default=None)
    detailed_time = field(default=None)

    @property
    def samples(self):
        return (
            self.refined_samples
            if self.refined_samples is not None
            else self.coarse_samples
        )

    def minimum(self, vios_eps = 1e-5):
        f = self.problem_instance.obj
        if self.refined_samples is not None:
            fs = np.array([f(x) for x in self.refined_samples if x is not None])
            if isinstance(self.problem_instance, BoxQP):
                maxvios = np.array([self.problem_instance.max_vios(x) for x in self.refined_samples if x is not None])
                feas = maxvios < vios_eps
                return min(fs[feas])
            return min(fs)
        return min([f(x) for x in self.coarse_samples if x is not None])

    @property
    def minimizer(self):
        best_i = 0
        f = self.problem_instance.obj
        for i in range(len(self.samples)):
            if f(self.samples[i]) < f(self.samples[best_i]):
                best_i = i
        return self.samples[best_i]

    @property
    def coarse_min(self):
        f = self.problem_instance.obj
        return min([f(x) for x in self.coarse_samples if x is not None])

    @property
    def percent_in_subspace(self):
        return len([x for x in self.samples if x is not None]) / len(self.samples)

    def succ_prob(self, tol=1e-3):
        num_samples = sum(self.sample_counts)
        num_succ = 0
        f = self.problem_instance.obj
        minimum = self.minimum(tol)
        for sample, count in zip(self.samples, self.sample_counts):
            if abs(f(sample) - minimum) < tol:
                num_succ += count
        return num_succ / num_samples

    def succ_prob_coarse(self, tol=1e-3):
        num_succ = 0
        f = self.problem_instance.obj
        for sample in self.coarse_samples:
            if sample is not None and abs(f(sample) - self.minimum) < tol:
                num_succ += 1
        return num_succ / len(self.samples)