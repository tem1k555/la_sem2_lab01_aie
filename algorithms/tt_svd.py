"""
TT-SVD алгоритм для преобразования плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    k = S.shape[0]
    r = k
    
    if S.data[0] == 0:
        return 1
    
    max_val = max(1e-12, 1e-8 * S.data[0])
    for j in range(k):
        if S.data[j] <= max_val:
            r = j
            break
    
    if delta > 0:
        s_sum = 0.0
        for j in range(r - 1, -1, -1):
            s_sum += S.data[j] ** 2
            if s_sum > delta ** 2:
                break
            r = j
    
    r = max(1, r)
    if max_rank is not None:
        r = min(r, max_rank)
    return r


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    m = backend.shape(matrix)[0]
    res = backend.zeros((m, rank))
    for i in range(m):
        for j in range(rank):
            backend.set_element(res, (i, j), backend.get_element(matrix, (i, j)))
    return res


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    n = backend.shape(matrix)[1]
    res = backend.zeros((rank, n))
    for i in range(rank):
        for j in range(n):
            backend.set_element(res, (i, j), backend.get_element(matrix, (i, j)))
    return res


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    res = backend.zeros((rank,))
    for i in range(rank):
        backend.set_element(res, (i,), backend.get_element(vector, (i,)))
    return res


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    n = backend.shape(matrix)[1]
    res = backend.zeros((rank, n))
    for i in range(rank):
        d = backend.get_element(diag_vec, (i,))
        for j in range(n):
            val = backend.get_element(matrix, (i, j))
            backend.set_element(res, (i, j), d * val)
    return res


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-15  # ← ИЗМЕНЕНО: было 1e-10
) -> TTTensor:
    """
    Преобразует плотный тензор в TT-формат с помощью SVD.
    
    Args:
        tensor: исходный плотный тензор
        backend: бэкенд для операций
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps: относительная точность усечения (по умолчанию 1e-15)
    
    Returns:
        TTTensor: тензор в TT-формате
    """
    C = backend.copy(tensor)
    d = len(tensor.shape)
    n = tensor.shape
    
    if d == 1:
        G1 = backend.reshape(C, (1, n[0], 1))
        return TTTensor([G1])
    
    norm_A = backend.norm(tensor)
    delta = (eps / math.sqrt(d - 1)) * norm_A if norm_A > 1e-30 else 0.0
    
    cores = []
    r = [1] * (d + 1)

    for k in range(1, d):
        C_k = backend.reshape(C, (r[k - 1] * n[k - 1], backend.size(C) // (r[k - 1] * n[k - 1])))
        U, S, Vt = backend.svd(C_k, full_matrices=False)
        
        r_k = _compute_truncated_rank(S, delta, max_rank)
        r[k] = r_k
        
        U_trunc = _truncate_columns(U, r_k, backend)
        G_k = backend.reshape(U_trunc, (r[k - 1], n[k - 1], r[k]))
        cores.append(G_k)
        
        S_trunc = _truncate_vector(S, r_k, backend)
        Vt_trunc = _truncate_rows(Vt, r_k, backend)
        C = _multiply_diag_matrix(S_trunc, Vt_trunc, r_k, backend)

    G_d = backend.reshape(C, (r[d - 1], n[d - 1], 1))
    cores.append(G_d)
    
    return TTTensor(cores)