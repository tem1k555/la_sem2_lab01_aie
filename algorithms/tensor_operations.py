"""
Базовые операции с TT-тензорами.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.tt_svd import tt_svd  # ★ Добавляем импорт


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
    
    # Конвертируем обратно в TT через SVD
    return tt_svd(full_sum, backend, max_rank=None, eps=1e-10)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """Умножение TT-тензора на скаляр."""
    if alpha == 1.0:
        return tt.copy()
    
    cores = [core.copy() for core in tt.cores]
    core0 = cores[0]
    r_prev, n_0, r_next = core0.shape
    
    new_data = [alpha * x for x in core0.data]
    cores[0] = DenseTensor([r_prev, n_0, r_next], new_data)
    
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """Поэлементное произведение (Адамара)."""
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    full1 = tt1.full()
    full2 = tt2.full()
    full_prod = DenseTensor(full1.shape, 
                           [a * b for a, b in zip(full1.data, full2.data)])
    
    # Конвертируем обратно в TT через SVD
    return tt_svd(full_prod, backend, max_rank=None, eps=1e-10)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """Скалярное произведение двух TT-тензоров."""
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    d = tt1.order
    cores1 = tt1.cores
    cores2 = tt2.cores
    
    # Z_1 = sum_{i_1} G_1^A[i_1]^T @ G_1^B[i_1]
    core1_0 = cores1[0]
    core2_0 = cores2[0]
    r1_1 = core1_0.shape[2]
    r2_1 = core2_0.shape[2]
    n_0 = core1_0.shape[1]
    
    Z_data = []
    for i in range(r1_1):
        for j in range(r2_1):
            val = 0.0
            for idx in range(n_0):
                val += (core1_0.data[0 * n_0 * r1_1 + idx * r1_1 + i] *
                        core2_0.data[0 * n_0 * r2_1 + idx * r2_1 + j])
            Z_data.append(val)
    
    Z = DenseTensor([r1_1, r2_1], Z_data)
    
    for k in range(1, d):
        core1 = cores1[k]
        core2 = cores2[k]
        r1_prev, n_k, r1_next = core1.shape
        r2_prev, _, r2_next = core2.shape
        
        new_Z_data = []
        for i in range(r1_next):
            for j in range(r2_next):
                val = 0.0
                for idx in range(n_k):
                    for p in range(r1_prev):
                        for q in range(r2_prev):
                            val += (core1.data[p * n_k * r1_next + idx * r1_next + i] *
                                    Z.data[p * r2_prev + q] *
                                    core2.data[q * n_k * r2_next + idx * r2_next + j])
                new_Z_data.append(val)
        
        Z = DenseTensor([r1_next, r2_next], new_Z_data)
    
    return Z.data[0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """Фробениусова норма TT-тензора."""
    return math.sqrt(tt_dot(tt, tt, backend))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """Норма разности: ||tt1 - tt2||_F."""
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    norm1_sq = tt_norm(tt1, backend) ** 2
    norm2_sq = tt_norm(tt2, backend) ** 2
    dot = tt_dot(tt1, tt2, backend)
    
    diff_sq = norm1_sq + norm2_sq - 2 * dot
    
    if diff_sq < 0 and diff_sq > -1e-10:
        diff_sq = 0.0
    
    return math.sqrt(diff_sq)