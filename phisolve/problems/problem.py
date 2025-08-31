import abc

import numpy as np


class Problem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def obj(self, x):
        pass

    @abc.abstractmethod
    def clone(self):
        pass

    @abc.abstractmethod
    def grad(self, x):
        pass

    @abc.abstractmethod
    def with_new_bounds(self, bounds):
        pass

    @property
    @abc.abstractmethod
    def bounds(self):
        pass

    @property
    @abc.abstractmethod
    def nvar(self):
        pass

    def with_trust_region(self, x, scaling_factor=1):
        """
        Returns a new problem with bounds divided by d around a point x
        """
        delta = 1 / scaling_factor
        curr_lower_bound, curr_upper_bound = self.bounds
        new_lower_bound = np.array(
            [max(curr_lower_bound[i] - (x[i]), -1 * delta) for i in range(self.nvar)]
        )
        new_upper_bound = np.array(
            [min(curr_upper_bound[i] - x[i], delta) for i in range(self.nvar)]
        )

        return self.with_new_bounds((new_lower_bound, new_upper_bound))
