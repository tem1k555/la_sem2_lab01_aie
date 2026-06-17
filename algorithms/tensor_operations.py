"""
Базовые операции с TT-тензорами.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """Поэлементное сложение двух TT-тензоров."""
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    d = tt1.order
    cores1 = tt1.cores
    cores2 = tt2.cores
    
    new_cores = []
    
    for k in range(d):
        core1 = cores1[k]
        core2 = cores2[k]
        r1_prev, n_k, r1_next = core1.shape
        r2_prev, _, r2_next = core2.shape
        
        if k == 0:
            new_r_prev = 1
            new_r_next = r1_next + r2_next
            
            new_data = []
            for i in range(n_k):
                # Берем срез core1[0, i, :]
                for j in range(r1_next):
                    new_data.append(core1.data[0 * n_k * r1_next + i * r1_next + j])
                # Берем срез core2[0, i, :]
                for j in range(r2_next):
                    new_data.append(core2.data[0 * n_k * r2_next + i * r2_next + j])
            
            new_core = DenseTensor([new_r_prev, n_k, new_r_next], new_data)
            
        elif k == d - 1:
            new_r_prev = r1_prev + r2_prev
            new_r_next = 1
            
            new_data = []
            for i in range(n_k):
                # Берем срез core1[:, i, 0]
                for j in range(r1_prev):
                    new_data.append(core1.data[j * n_k * r1_next + i * r1_next + 0])
                # Берем срез core2[:, i, 0]
                for j in range(r2_prev):
                    new_data.append(core2.data[j * n_k * r2_next + i * r2_next + 0])
            
            new_core = DenseTensor([new_r_prev, n_k, new_r_next], new_data)
            
        else:
            
            new_r_prev = r1_prev + r2_prev
            new_r_next = r1_next + r2_next
            
            new_data = []
            for i in range(n_k):
                for p in range(new_r_prev):
                    for q in range(new_r_next):
                        if p < r1_prev and q < r1_next:
                            # Блок A
                            val = core1.data[p * n_k * r1_next + i * r1_next + q]
                        elif p >= r1_prev and q >= r1_next:
                            # Блок B
                            p2 = p - r1_prev
                            q2 = q - r1_next
                            val = core2.data[p2 * n_k * r2_next + i * r2_next + q2]
                        else:
                            val = 0.0
                        new_data.append(val)
            
            new_core = DenseTensor([new_r_prev, n_k, new_r_next], new_data)
        
        new_cores.append(new_core)
    
    return TTTensor(new_cores)

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
    """Поэлементное произведение (Адамара). G_k^C[i] = G_k^A[i] ⊗ G_k^B[i]"""
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    d = tt1.order
    cores1 = tt1.cores
    cores2 = tt2.cores
    
    new_cores = []
    
    for k in range(d):
        core1 = cores1[k]
        core2 = cores2[k]
        r1_prev, n_k, r1_next = core1.shape
        r2_prev, _, r2_next = core2.shape
        
        new_r_prev = r1_prev * r2_prev
        new_r_next = r1_next * r2_next
        
        new_data = []
        for i in range(n_k):
            for p1 in range(r1_prev):
                for p2 in range(r2_prev):
                   
                    p = p1 * r2_prev + p2
                    for q1 in range(r1_next):
                        for q2 in range(r2_next):
                            q = q1 * r2_next + q2
                            val = (core1.data[p1 * n_k * r1_next + i * r1_next + q1] *
                                   core2.data[p2 * n_k * r2_next + i * r2_next + q2])
                            new_data.append(val)
        
        new_core = DenseTensor([new_r_prev, n_k, new_r_next], new_data)
        new_cores.append(new_core)
    
    return TTTensor(new_cores)

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