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
    """
    Возвращает ранг усечения по сингулярным значениям.
    """
    if S.size == 0:
        return 1

    threshold = max(10 ** -12, abs(S.data[0]) * 10 ** -8)
    rank = 0
    for value in S.data:
        if abs(value) > threshold:
            rank += 1

    if rank == 0:
        rank = 1

    r = rank
    for curr_rank in range(1, rank + 1):
        tail = sum(S.data[i] ** 2 for i in range(curr_rank, rank))
        if tail <= delta ** 2:
            r = curr_rank
            break

    if max_rank is not None:
        r = min(r, max_rank)
    return max(1, r)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.
    """
    rows, cols = matrix.shape
    data = []
    for i in range(rows):
        for j in range(rank):
            data.append(matrix.data[i * cols + j])
    return DenseTensor((rows, rank), data)


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.
    """
    cols = matrix.shape[1]
    return DenseTensor((rank, cols), matrix.data[:rank * cols])


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.
    """
    return DenseTensor((rank,), vector.data[:rank])


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу.
    """
    return backend.matmul(backend.diag(diag_vec), matrix)


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.
    """
    if tensor.ndim == 1:
        return TTTensor([tensor.reshape((1, tensor.shape[0], 1))])

    cores = []
    prev_r = 1
    C = tensor.copy()
    frob_norm = tensor.norm()
    delta = eps * frob_norm / math.sqrt(tensor.ndim - 1) if frob_norm > 10 ** -30 else 0
    
    for k in range(tensor.ndim - 1):
        rows = prev_r * tensor.shape[k]
        C = C.reshape((rows, C.size // rows))
        U, singular_values, V = backend.svd(C)
        r = _compute_truncated_rank(singular_values, delta, max_rank)
        U = _truncate_columns(U, r, backend)
        cores.append(U.reshape((prev_r, tensor.shape[k], r)))
        singular_values = _truncate_vector(singular_values, r, backend)
        V = _truncate_rows(V, r, backend)
        C = _multiply_diag_matrix(singular_values, V, r, backend)
        prev_r = r
        
    cores.append(C.reshape((prev_r, tensor.shape[-1], 1)))
    return TTTensor(cores)