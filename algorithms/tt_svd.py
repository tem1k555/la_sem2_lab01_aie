"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.
    """
    d = tensor.ndim
    
    # Если тензор порядка 1
    if d == 1:
        core = tensor.reshape([1, tensor.shape[0], 1])
        return TTTensor([core])
    
    norm_tensor = tensor.norm()
    
    # Если норма близка к нулю
    if norm_tensor < 1e-30:
        cores = []
        for k in range(d):
            r_prev = 1 if k == 0 else 1
            r_next = 1 if k == d - 1 else 1
            core = DenseTensor.zeros([r_prev, tensor.shape[k], r_next])
            cores.append(core)
        return TTTensor(cores)
    
    delta = (eps / math.sqrt(d - 1)) * norm_tensor if d > 1 else 0.0
    
    # Инициализация
    C = tensor.copy()
    cores = []
    r_prev = 1
    
    # Основной цикл
    for k in range(d - 1):
        # Левый размер: r_prev * n_k
        left_dim = r_prev * tensor.shape[k]
        # Правый размер: произведение оставшихся размерностей
        right_dim = 1
        for i in range(k + 1, tensor.ndim):
            right_dim *= tensor.shape[i]
        
        # Разворачиваем C в матрицу
        # C имеет размер (r_prev, n_k, n_{k+1}, ..., n_{d-1})
        # Нужно получить матрицу размера (r_prev * n_k, right_dim)
        
        # Если C уже матрица или вектор
        if C.ndim == 1:
            # Вектор -> матрица 1 x n
            matrix = C.reshape([1, C.size])
        elif C.ndim == 2:
            # Уже матрица, просто перестраиваем
            matrix = C.reshape([left_dim, right_dim])
        else:
            # Тензор > 2D: используем left_unfolding
            # C имеет размер (r_prev, n_k, n_{k+1}, ...)
            # left_unfolding(0) даёт (r_prev * n_k, right_dim)
            matrix = C.left_unfolding(0)
        
        # SVD
        U, S, Vt = backend.svd(matrix, full_matrices=False)
        
        # Выбираем ранг
        rank = _compute_truncated_rank(S, delta, max_rank)
        
        # Усекаем U
        U_trunc = _truncate_columns(U, rank, backend)
        
        # Формируем ядро G_k: перестраиваем U_trunc в (r_prev, n_k, rank)
        n_k = tensor.shape[k]
        G_k_data = []
        for i in range(r_prev):
            for j in range(n_k):
                for l in range(rank):
                    G_k_data.append(U_trunc.data[(i * n_k + j) * rank + l])
        
        G_k = DenseTensor([r_prev, n_k, rank], G_k_data)
        cores.append(G_k)
        
        # Обновляем C для следующего шага
        Sigma = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)
        
        # C = diag(Sigma) @ Vt_trunc
        new_data = []
        for i in range(rank):
            for j in range(Vt_trunc.shape[1]):
                new_data.append(Sigma.data[i] * Vt_trunc.data[i * Vt_trunc.shape[1] + j])
        
        C = DenseTensor([rank, Vt_trunc.shape[1]], new_data)
        r_prev = rank
    
    # Последнее ядро
    G_last = C.reshape([r_prev, tensor.shape[-1], 1])
    cores.append(G_last)
    
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """Возвращает ранг усечения по сингулярным значениям."""
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