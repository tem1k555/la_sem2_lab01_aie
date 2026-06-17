"""
Приведение TT-тензора в канонические формы.
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Лево-каноническая форма.
    """
    if tt.order == 1:
        return tt.copy()
    
    cores = [core.copy() for core in tt.cores]
    d = tt.order
    
    for k in range(d - 1):
        core = cores[k]
        r_prev, n_k, r_next = core.shape
        
        # Разворачиваем в матрицу (r_prev * n_k) x r_next
        G = core.reshape([r_prev * n_k, r_next])
        
        # QR-разложение: G = Q @ R
        Q, R = backend.qr(G)
        
        # Заменяем G_k на Q (перестроенный в тензор)
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
        
        # Поглощаем R в следующее ядро
        next_core = cores[k + 1]
        r_prev_next, n_next, r_next_next = next_core.shape
        
        # R имеет размер (r_next, r_next)
        # Нужно умножить R на G_{k+1}: G_{k+1}[i] = R @ G_{k+1}[i]
        new_next_data = []
        for i in range(n_next):
            for p in range(r_next):
                for q in range(r_next_next):
                    val = 0.0
                    for t in range(r_next):
                        val += R.data[p * r_next + t] * next_core.data[t * n_next * r_next_next + i * r_next_next + q]
                    new_next_data.append(val)
        
        cores[k + 1] = DenseTensor([r_next, n_next, r_next_next], new_next_data)
    
    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Право-каноническая форма.
    """
    if tt.order == 1:
        return tt.copy()
    
    cores = [core.copy() for core in tt.cores]
    d = tt.order
    
    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_prev, n_k, r_next = core.shape
        
        # Разворачиваем в матрицу r_prev x (n_k * r_next)
        G = core.reshape([r_prev, n_k * r_next])
        
        # Выполняем RQ-разложение: G = R @ Q
        # Используем QR от транспонированной: G^T = Q_t @ R_t
        # Тогда G = R_t^T @ Q_t^T
        Gt = DenseTensor([n_k * r_next, r_prev], 
                        [G.data[i * (n_k * r_next) + j] for j in range(r_prev) for i in range(n_k * r_next)])
        
        Q_t, R_t = backend.qr(Gt)
        
        # Q = Q_t^T, R = R_t^T
        Q_data = [Q_t.data[j * r_prev + i] for i in range(r_prev) for j in range(n_k * r_next)]
        Q = DenseTensor([r_prev, n_k * r_next], Q_data)
        
        R_data = [R_t.data[j * r_prev + i] for i in range(r_prev) for j in range(r_prev)]
        R = DenseTensor([r_prev, r_prev], R_data)
        
        # Заменяем G_k на Q (перестроенный в тензор)
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
        
        # Поглощаем R в предыдущее ядро
        prev_core = cores[k - 1]
        r_prev_prev, n_prev, r_prev_next = prev_core.shape
        
        # G_{k-1}[i] = G_{k-1}[i] @ R
        new_prev_data = []
        for i in range(n_prev):
            for p in range(r_prev_prev):
                for q in range(r_prev):
                    val = 0.0
                    for t in range(r_prev):
                        val += prev_core.data[p * n_prev * r_prev_next + i * r_prev_next + t] * R.data[t * r_prev + q]
                    new_prev_data.append(val)
        
        cores[k - 1] = DenseTensor([r_prev_prev, n_prev, r_prev], new_prev_data)
    
    return TTTensor(cores)