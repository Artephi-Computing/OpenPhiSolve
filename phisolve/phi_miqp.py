import numpy as np
import scipy as sp
from jax import vmap, jit
import jax.numpy as jnp

import time
from attr import define, field

import numpy as np
import scipy as sp
from phisolve.problems import Problem, QUBO, MIQP
from phisolve.backends import Backend
from phisolve.refiners import Refiner
from phisolve.response import Response
import jax.numpy as jnp
from phisolve import QIHD, PDQP

@define
class PhiMIQP:
    problem_instance: Problem = field()
    backend: Backend = field(default=QIHD())
    refiner: Refiner = field(default=PDQP())
    remove_diagonal: bool = field(default=True)
    compiled_problem_instance: MIQP = field(default=None)

    def __attrs_post_init__(self):
        if not isinstance(self.problem_instance, MIQP):
            kwargs = self.problem_instance.to_dict()
            if isinstance(self.problem_instance, QUBO):
                kwargs['n_binary_vars'] = self.problem_instance.Q.shape[0]
            self.problem_instance = MIQP(**kwargs)
    
    def compile(self, N):
        Q, w = self.problem_instance.calc_Q_w_affine()
        A, b, C, d = self.problem_instance.calc_A_b_C_d_affine()
        if isinstance(Q, sp.sparse.spmatrix):
            Q[:N, :N].setdiag(0)
        else:
            np.fill_diagonal(Q[:N, :N], 0)

        return Q, w, A, b, C, d


    def solve(
        self,
        if_refine=True,
    ):
        det_time = dict()
        if self.remove_diagonal == True:
            self.remove_diagonal = self.problem_instance.nvar
        elif self.remove_diagonal == False:
            self.remove_diagonal = 0

        Q, w, A, b, C, d = self.compile(self.remove_diagonal)
        QIHD_n_bin_var = self.problem_instance.nvar  # In general, treating BoxQP as QUBO works better in QIHD.
        self.compiled_problem_instance = MIQP(Q=Q, 
                                              w=w, 
                                              A=A, 
                                              b=b, 
                                              C=C, 
                                              d=d,
                                              n_binary_vars=QIHD_n_bin_var)

        if self.problem_instance.nvar == self.problem_instance.n_binary_vars:
            # For QUBO problem: override if if_refine == True
            if_refine = False 

        start_time = time.time()
        samples = self.backend.generate_samples(self.compiled_problem_instance) # use backend to solve the compiled MIQP (with the same solutions)
        samples = vmap(self.problem_instance.affine_trans)(samples)
        samples, sample_counts = np.unique(samples, axis=0, return_counts=True)

        det_time["QIHD_Time"] = time.time() - start_time

        if if_refine:
            if self.refiner.vmappable:
                refined_samples, refine_time = self._refine_vmapped(samples)
            else:
                refined_samples, refine_time = self._refine_batched(samples)
            det_time["Refinement"] = refine_time
            return Response(self.problem_instance, samples, sample_counts, refined_samples, detailed_time=det_time)
        else:
            return Response(self.problem_instance, samples, sample_counts, samples, detailed_time=det_time)

    def _refine_batched(self, samples, problem=None):
        refine_start_time = time.time()
        if problem == None:
            problem = self.problem_instance
        n_bin = problem.n_binary_vars
        refined_samples = np.zeros_like(samples)
        bin_dict = dict()
        bin_indices = dict()
        for idx, sample in enumerate(samples):
            binval = tuple(sample[:n_bin])
            item = sample[n_bin:]
            if binval in bin_dict:
                bin_dict[binval].append(item)
                bin_indices[binval].append(idx)
            else:
                bin_dict[binval] = [item]
                bin_indices[binval] = [idx]
        for binval in bin_dict:
            binind = bin_indices[binval]
            lcqp_instance = problem.fix_binary_values(binval)
            batched_samples = jnp.stack(bin_dict[binval])
            batch_refined_sample = self.refiner.refine(samples=batched_samples, problem=lcqp_instance)
            for i in range(len(binind)):
                refined_samples[binind[i]] = np.concatenate((np.array(binval), np.array(batch_refined_sample[i])))
        refine_time = time.time() - refine_start_time
        return refined_samples, refine_time

    def _refine_vmapped(self, samples, problem=None):
        refine_start_time = time.time()
        if problem == None:
            problem = self.problem_instance
        n_bin = problem.n_binary_vars
        
        def _refine_one_sample(sample):
            binval = sample[:n_bin]
            lcqp_instance = problem.fix_binary_values(binval)
            refined_sample = self.refiner.refine(samples=sample[None, n_bin:], problem=lcqp_instance)
            return jnp.concatenate((jnp.array(binval), refined_sample[0]))
        
        refined_samples = jit(vmap(_refine_one_sample))(samples)

        refine_time = time.time() - refine_start_time
        return refined_samples, refine_time
    