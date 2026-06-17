"""
Базовые операции с TT-тензорами.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.tt_svd import tt_svd


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """Поэлементное сложение двух TT-тензоров."""
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    full1 = tt1.full()
    full2 = tt2.full()
    full_sum = full1 + full2
    
    # ★ Максимальная точность
    return tt_svd(full_sum, backend, max_rank=None, eps=1e-20)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """Умножение TT-тензора на скаляр."""
    if alpha == 1.0:
        return tt.copy()
    
    cores = [backend.copy(core) for core in tt.cores]
    cores[0] = backend.scale(cores[0], alpha)
    
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """Поэлементное произведение (Адамара) двух TT-тензоров."""
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    full1 = tt1.full()
    full2 = tt2.full()
    full_prod = DenseTensor(full1.shape, 
                           [a * b for a, b in zip(full1.data, full2.data)])
    
    # ★ Максимальная точность
    return tt_svd(full_prod, backend, max_rank=None, eps=1e-20)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """Скалярное произведение двух TT-тензоров."""
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    d = tt1.order
    Z = backend.ones((1, 1))
    
    for k in range(d):
        rA_in, n, rA_out = backend.shape(tt1.cores[k])
        rB_in, _, rB_out = backend.shape(tt2.cores[k])
        Z_next = backend.zeros((rA_out, rB_out))
        
        for i in range(n):
            GA_i = backend.zeros((rA_in, rA_out))
            for r1 in range(rA_in):
                for r2 in range(rA_out):
                    backend.set_element(GA_i, (r1, r2), backend.get_element(tt1.cores[k], (r1, i, r2)))
            
            GB_i = backend.zeros((rB_in, rB_out))
            for r1 in range(rB_in):
                for r2 in range(rB_out):
                    backend.set_element(GB_i, (r1, r2), backend.get_element(tt2.cores[k], (r1, i, r2)))
            
            term = backend.matmul(backend.transpose(GA_i), backend.matmul(Z, GB_i))
            Z_next = backend.add(Z_next, term)
        
        Z = Z_next
    
    return backend.get_element(Z, (0, 0))


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """Фробениусова норма TT-тензора."""
    return math.sqrt(max(0.0, tt_dot(tt, tt, backend)))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """Норма разности двух TT-тензоров."""
    n1 = tt_dot(tt1, tt1, backend)
    n2 = tt_dot(tt2, tt2, backend)
    dot = tt_dot(tt1, tt2, backend)
    return math.sqrt(max(0.0, n1 + n2 - 2.0 * dot))