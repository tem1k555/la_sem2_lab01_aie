"""
Приведение TT-тензора в канонические формы.
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


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


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    if tt.order == 1:
        return tt.copy()
    
    cores = [backend.copy(core) for core in tt.cores]
    d = len(cores)
    
    for k in range(d - 1):
        r_k, n_k, r_kp1 = backend.shape(cores[k])
        G_unf = backend.reshape(cores[k], (r_k * n_k, r_kp1))
        
        m, n = backend.shape(G_unf)
        if m >= n:
            Q, R = backend.qr(G_unf)
        else:
            U, S, Vt = backend.svd(G_unf, full_matrices=False)
            Q = U
            R = _multiply_diag_matrix(S, Vt, backend.shape(S)[0], backend)
        
        new_r = backend.shape(Q)[1]
        cores[k] = backend.reshape(Q, (r_k, n_k, new_r))
        
        r_next, n_next, r_nextp1 = backend.shape(cores[k + 1])
        G_next_unf = backend.reshape(cores[k + 1], (r_next, n_next * r_nextp1))
        cores[k + 1] = backend.reshape(backend.matmul(R, G_next_unf), (new_r, n_next, r_nextp1))
    
    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    if tt.order == 1:
        return tt.copy()
    
    cores = [backend.copy(core) for core in tt.cores]
    d = len(cores)
    
    for k in range(d - 1, 0, -1):
        r_k, n_k, r_kp1 = backend.shape(cores[k])
        G_unf = backend.reshape(cores[k], (r_k, n_k * r_kp1))
        
        G_unf_T = backend.transpose(G_unf)
        m, n = backend.shape(G_unf_T)
        
        if m >= n:
            Q_T, R_T = backend.qr(G_unf_T)
            Q = backend.transpose(Q_T)
            R = backend.transpose(R_T)
        else:
            U, S, Vt = backend.svd(G_unf_T, full_matrices=False)
            Q_T = U
            R_T = _multiply_diag_matrix(S, Vt, backend.shape(S)[0], backend)
            Q = backend.transpose(Q_T)
            R = backend.transpose(R_T)
        
        new_r = backend.shape(Q)[0]
        cores[k] = backend.reshape(Q, (new_r, n_k, r_kp1))
        
        r_prev, n_prev, r_p = backend.shape(cores[k - 1])
        G_prev_unf = backend.reshape(cores[k - 1], (r_prev * n_prev, r_p))
        cores[k - 1] = backend.reshape(backend.matmul(G_prev_unf, R), (r_prev, n_prev, new_r))
    
    return TTTensor(cores)