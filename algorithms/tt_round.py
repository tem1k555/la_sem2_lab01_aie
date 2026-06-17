"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    TT-округление: уменьшение рангов с контролем точности.
    """
    if tt.order == 1:
        return tt.copy()
    
    # 1. Правый проход: право-каноническая форма
    tt_right = right_canonicalize(tt, backend)
    cores = [core.copy() for core in tt_right.cores]
    d = tt.order
    
    # Норма первого ядра (вся норма в нём после right_canonicalize)
    norm_first = cores[0].norm()
    
    if norm_first < 1e-30:
        new_cores = []
        for k in range(d):
            r_prev = 1 if k == 0 else 1
            r_next = 1 if k == d - 1 else 1
            core = DenseTensor.zeros([r_prev, tt.shape[k], r_next])
            new_cores.append(core)
        return TTTensor(new_cores)
    
    delta = (eps / math.sqrt(d - 1)) * norm_first if d > 1 else 0.0
    
    r_prev = 1
    
    # 2. Левый проход: SVD-усечение
    for k in range(d - 1):
        core = cores[k]
        r_prev_core, n_k, r_next_core = core.shape
        
        # Разворачиваем в матрицу (r_prev * n_k) x r_next
        G = core.reshape([r_prev * n_k, r_next_core])
        
        # SVD
        U, S, Vt = backend.svd(G, full_matrices=False)
        
        # Вычисляем новый ранг
        new_rank = _compute_rank(S, delta, max_rank)
        
        # Усекаем U
        U_trunc = _truncate_columns(U, new_rank, backend)
        
        # Формируем новое ядро
        new_core = U_trunc.reshape([r_prev, n_k, new_rank])
        cores[k] = new_core
        
        # Поглощаем остаток в следующее ядро
        Sigma = _truncate_vector(S, new_rank, backend)
        Vt_trunc = _truncate_rows(Vt, new_rank, backend)
        
        # absorbed = diag(Sigma) @ Vt_trunc
        absorbed_data = []
        for i in range(new_rank):
            for j in range(Vt_trunc.shape[1]):
                absorbed_data.append(Sigma.data[i] * Vt_trunc.data[i * Vt_trunc.shape[1] + j])
        
        absorbed = DenseTensor([new_rank, Vt_trunc.shape[1]], absorbed_data)
        
        # Поглощаем в следующее ядро
        next_core = cores[k + 1]
        r_prev_next, n_next, r_next_next = next_core.shape
        
        # absorbed имеет размер (new_rank, n_next * r_next_next)
        # Перестраиваем absorbed в (new_rank, n_next, r_next_next)
        if absorbed.size == new_rank * n_next * r_next_next:
            absorbed_reshaped = absorbed.reshape([new_rank, n_next, r_next_next])
        else:
            # Если размерности не совпадают, нужно перестроить
            # Создаём новый тензор
            absorbed_flat = absorbed.data
            new_absorbed_data = []
            for i in range(new_rank):
                for j in range(n_next):
                    for q in range(r_next_next):
                        idx = i * n_next * r_next_next + j * r_next_next + q
                        if idx < len(absorbed_flat):
                            new_absorbed_data.append(absorbed_flat[idx])
                        else:
                            new_absorbed_data.append(0.0)
            absorbed_reshaped = DenseTensor([new_rank, n_next, r_next_next], new_absorbed_data)
        
        # Умножаем absorbed_reshaped на next_core по первой моде
        # G_{k+1}[i] = absorbed_reshaped[:, i, :] @ G_{k+1}[i]
        # Но на самом деле: new_next_core = absorbed_reshaped * next_core (свёртка по r_prev)
        new_next_data = []
        for i in range(n_next):
            for p in range(new_rank):
                for q in range(r_next_next):
                    val = 0.0
                    for t in range(r_prev_next):
                        val += (absorbed_reshaped.data[p * n_next * r_next_next + i * r_next_next + t] *
                                next_core.data[t * n_next * r_next_next + i * r_next_next + q])
                    new_next_data.append(val)
        
        cores[k + 1] = DenseTensor([new_rank, n_next, r_next_next], new_next_data)
        r_prev = new_rank
    
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """Возвращает ранг усечения."""
    if S.size == 0:
        return 1
    
    sigma1 = S.data[0] if S.data else 0.0
    numerical_rank = len(S.data)
    
    for i, sigma in enumerate(S.data):
        if sigma <= max(1e-12, 1e-8 * sigma1):
            numerical_rank = i
            break
    
    truncated_rank = numerical_rank
    
    if delta > 0:
        for r in range(numerical_rank, 0, -1):
            dropped_sq = sum(sigma * sigma for sigma in S.data[r:])
            if dropped_sq <= delta * delta:
                truncated_rank = r
                break
    
    if max_rank is not None:
        truncated_rank = min(truncated_rank, max_rank)
    
    return max(1, truncated_rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """Возвращает матрицу из первых rank столбцов."""
    if rank >= matrix.shape[1]:
        return matrix.copy()
    
    m, n = matrix.shape
    data = []
    for i in range(m):
        for j in range(rank):
            data.append(matrix.data[i * n + j])
    
    return DenseTensor([m, rank], data)


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """Возвращает матрицу из первых rank строк."""
    if rank >= matrix.shape[0]:
        return matrix.copy()
    
    m, n = matrix.shape
    data = []
    for i in range(rank):
        for j in range(n):
            data.append(matrix.data[i * n + j])
    
    return DenseTensor([rank, n], data)


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """Возвращает вектор из первых rank элементов."""
    if rank >= vector.size:
        return vector.copy()
    
    return DenseTensor([rank], vector.data[:rank])