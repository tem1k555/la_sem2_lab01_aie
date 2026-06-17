"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    k = S.shape[0]
    if S.data[0] == 0:
        return 1
    
    r = k
    if delta > 0:
        s_sum = 0.0
        for j in range(k - 1, -1, -1):
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


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    tt = right_canonicalize(tt, backend)
    d = tt.order
    norm_A = backend.norm(tt.cores[0])
    delta = (eps / math.sqrt(d - 1)) * norm_A if d > 1 else 0.0
    cores = [backend.copy(c) for c in tt.cores]

    for k in range(d - 1):
        r_k, n_k, r_kp1 = backend.shape(cores[k])
        G_unf = backend.reshape(cores[k], (r_k * n_k, r_kp1))
        U, S, Vt = backend.svd(G_unf, full_matrices=False)
        
        new_r = _compute_rank(S, delta, max_rank)
        U_tr = _truncate_columns(U, new_r, backend)
        cores[k] = backend.reshape(U_tr, (r_k, n_k, new_r))
        
        S_tr = _truncate_vector(S, new_r, backend)
        Vt_tr = _truncate_rows(Vt, new_r, backend)
        M = _multiply_diag_matrix(S_tr, Vt_tr, new_r, backend)
        
        r_next, n_next, r_nextp1 = backend.shape(cores[k + 1])
        G_next_unf = backend.reshape(cores[k + 1], (r_next, n_next * r_nextp1))
        cores[k + 1] = backend.reshape(backend.matmul(M, G_next_unf), (new_r, n_next, r_nextp1))

    return TTTensor(cores)