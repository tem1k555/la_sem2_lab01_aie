"""
Приведение TT-тензора в канонические формы.
ВЕРСИЯ: простая канонизация через SVD с сохранением формы
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Лево-каноническая форма.
    Все ядра кроме последнего лево-ортогональны.
    """
    if tt.order == 1:
        return tt.copy()
    
    cores = [core.copy() for core in tt.cores]
    d = tt.order
    
    for k in range(d - 1):
        core = cores[k]
        r_prev, n_k, r_next = core.shape
        
        if r_prev * n_k == 0 or r_next == 0:
            continue
        
        G = core.reshape([r_prev * n_k, r_next])
        
        try:
            U, S, Vt = backend.svd(G, full_matrices=False)
            Q = U
        except Exception:
            Q, _ = backend.qr(G)
        
        # Обрезаем или дополняем Q до нужного размера
        if Q.shape[1] != r_next:
            if Q.shape[1] < r_next:
                new_data = []
                for i in range(Q.shape[0]):
                    for j in range(r_next):
                        new_data.append(Q.data[i * Q.shape[1] + j] if j < Q.shape[1] else 0.0)
                Q = DenseTensor([Q.shape[0], r_next], new_data)
            else:
                new_data = []
                for i in range(Q.shape[0]):
                    for j in range(r_next):
                        new_data.append(Q.data[i * Q.shape[1] + j])
                Q = DenseTensor([Q.shape[0], r_next], new_data)
        
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
    
    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Право-каноническая форма.
    Все ядра кроме первого право-ортогональны.
    """
    if tt.order == 1:
        return tt.copy()
    
    cores = [core.copy() for core in tt.cores]
    d = tt.order
    
    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_prev, n_k, r_next = core.shape
        
        if r_prev == 0 or n_k * r_next == 0:
            continue
        
        G = core.reshape([r_prev, n_k * r_next])
        
        try:
            Gt_data = []
            for j in range(r_prev):
                for i in range(n_k * r_next):
                    Gt_data.append(G.data[i * r_prev + j])
            Gt = DenseTensor([n_k * r_next, r_prev], Gt_data)
            
            U, S, Vt = backend.svd(Gt, full_matrices=False)
            
            Q_data = []
            for i in range(r_prev):
                for j in range(n_k * r_next):
                    Q_data.append(U.data[j * U.shape[1] + i] if j < U.shape[0] and i < U.shape[1] else 0.0)
            Q = DenseTensor([r_prev, n_k * r_next], Q_data)
            
        except Exception:
            Gt_data = []
            for j in range(r_prev):
                for i in range(n_k * r_next):
                    Gt_data.append(G.data[i * r_prev + j])
            Gt = DenseTensor([n_k * r_next, r_prev], Gt_data)
            
            Q_t, _ = backend.qr(Gt)
            
            Q_data = []
            for i in range(r_prev):
                for j in range(n_k * r_next):
                    Q_data.append(Q_t.data[j * Q_t.shape[1] + i] if j < Q_t.shape[0] and i < Q_t.shape[1] else 0.0)
            Q = DenseTensor([r_prev, n_k * r_next], Q_data)
        
        # Обрезаем или дополняем Q до нужного размера
        if Q.shape[1] != n_k * r_next:
            if Q.shape[1] < n_k * r_next:
                new_data = []
                for i in range(Q.shape[0]):
                    for j in range(n_k * r_next):
                        new_data.append(Q.data[i * Q.shape[1] + j] if j < Q.shape[1] else 0.0)
                Q = DenseTensor([Q.shape[0], n_k * r_next], new_data)
            else:
                new_data = []
                for i in range(Q.shape[0]):
                    for j in range(n_k * r_next):
                        new_data.append(Q.data[i * Q.shape[1] + j])
                Q = DenseTensor([Q.shape[0], n_k * r_next], new_data)
        
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
    
    return TTTensor(cores)