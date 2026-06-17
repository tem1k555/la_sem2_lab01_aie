"""
Приведение TT-тензора в канонические формы.
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
            
            # Q = U
            Q = U
            
            # R = diag(S) @ Vt
            R_data = []
            for i in range(len(S.data)):
                for j in range(Vt.shape[1]):
                    R_data.append(S.data[i] * Vt.data[i * Vt.shape[1] + j])
            R = DenseTensor([len(S.data), Vt.shape[1]], R_data)
            
        except Exception:
            # Если SVD упал, используем QR
            Q, R = backend.qr(G)
        
        if Q.shape[0] != r_prev * n_k or Q.shape[1] != r_next:
            if Q.shape[1] < r_next:
                # Дополняем Q нулевыми столбцами
                new_Q_data = []
                for i in range(r_prev * n_k):
                    for j in range(r_next):
                        if j < Q.shape[1]:
                            new_Q_data.append(Q.data[i * Q.shape[1] + j])
                        else:
                            new_Q_data.append(0.0)
                Q = DenseTensor([r_prev * n_k, r_next], new_Q_data)
            else:
                # Усекаем Q
                new_Q_data = []
                for i in range(r_prev * n_k):
                    for j in range(r_next):
                        new_Q_data.append(Q.data[i * Q.shape[1] + j])
                Q = DenseTensor([r_prev * n_k, r_next], new_Q_data)
        
        if R.shape[0] != r_next:
            if R.shape[0] < r_next:
                new_R_data = []
                for i in range(r_next):
                    for j in range(R.shape[1]):
                        if i < R.shape[0]:
                            new_R_data.append(R.data[i * R.shape[1] + j])
                        else:
                            new_R_data.append(0.0)
                R = DenseTensor([r_next, R.shape[1]], new_R_data)
            else:
                new_R_data = []
                for i in range(r_next):
                    for j in range(R.shape[1]):
                        new_R_data.append(R.data[i * R.shape[1] + j])
                R = DenseTensor([r_next, R.shape[1]], new_R_data)
        
        # Заменяем G_k на Q
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
        
        next_core = cores[k + 1]
        _, n_next, r_next_next = next_core.shape
        
        # Убеждаемся, что R имеет правильную форму для умножения
        if R.shape[1] != r_next:
            # R должен быть (r_next, r_next)
            if R.shape[1] < r_next:
                new_R_data = []
                for i in range(r_next):
                    for j in range(r_next):
                        if i < R.shape[0] and j < R.shape[1]:
                            new_R_data.append(R.data[i * R.shape[1] + j])
                        else:
                            new_R_data.append(0.0 if i != j else 1.0)
                R = DenseTensor([r_next, r_next], new_R_data)
            else:
                new_R_data = []
                for i in range(r_next):
                    for j in range(r_next):
                        new_R_data.append(R.data[i * R.shape[1] + j])
                R = DenseTensor([r_next, r_next], new_R_data)
        
        # G_{k+1}[i] = R @ G_{k+1}[i]
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
            # G^T = U @ diag(S) @ Vt
            Gt = DenseTensor([n_k * r_next, r_prev], 
                           [G.data[i * r_prev + j] for j in range(r_prev) for i in range(n_k * r_next)])
            U, S, Vt = backend.svd(Gt, full_matrices=False)
            
            # Q = U^T (для G)
            Q_data = []
            for i in range(r_prev):
                for j in range(n_k * r_next):
                    if j < U.shape[0] and i < U.shape[1]:
                        Q_data.append(U.data[j * U.shape[1] + i])
                    else:
                        Q_data.append(0.0)
            Q = DenseTensor([r_prev, n_k * r_next], Q_data)
            
            # R = Vt^T @ diag(S) (для G)
            R_data = []
            for i in range(r_prev):
                for j in range(r_prev):
                    if i < Vt.shape[0] and j < Vt.shape[1]:
                        R_data.append(Vt.data[j * Vt.shape[1] + i] * S.data[i])
                    else:
                        R_data.append(0.0)
            R = DenseTensor([r_prev, r_prev], R_data)
            
        except Exception:
            # Если SVD упал, используем QR
            Gt = DenseTensor([n_k * r_next, r_prev], 
                           [G.data[i * r_prev + j] for j in range(r_prev) for i in range(n_k * r_next)])
            Q_t, R_t = backend.qr(Gt)
            
            Q_data = []
            for i in range(r_prev):
                for j in range(n_k * r_next):
                    if j < Q_t.shape[0] and i < Q_t.shape[1]:
                        Q_data.append(Q_t.data[j * Q_t.shape[1] + i])
                    else:
                        Q_data.append(0.0)
            Q = DenseTensor([r_prev, n_k * r_next], Q_data)
            
            R_data = []
            for i in range(r_prev):
                for j in range(r_prev):
                    if i < R_t.shape[0] and j < R_t.shape[1]:
                        R_data.append(R_t.data[j * R_t.shape[1] + i])
                    else:
                        R_data.append(0.0)
            R = DenseTensor([r_prev, r_prev], R_data)
        
        if Q.shape[0] != r_prev or Q.shape[1] != n_k * r_next:
            if Q.shape[1] < n_k * r_next:
                new_Q_data = []
                for i in range(r_prev):
                    for j in range(n_k * r_next):
                        if j < Q.shape[1]:
                            new_Q_data.append(Q.data[i * Q.shape[1] + j])
                        else:
                            new_Q_data.append(0.0)
                Q = DenseTensor([r_prev, n_k * r_next], new_Q_data)
            else:
                new_Q_data = []
                for i in range(r_prev):
                    for j in range(n_k * r_next):
                        new_Q_data.append(Q.data[i * Q.shape[1] + j])
                Q = DenseTensor([r_prev, n_k * r_next], new_Q_data)
        

        if R.shape[0] != r_prev or R.shape[1] != r_prev:
            if R.shape[0] < r_prev or R.shape[1] < r_prev:
                new_R_data = []
                for i in range(r_prev):
                    for j in range(r_prev):
                        if i < R.shape[0] and j < R.shape[1]:
                            new_R_data.append(R.data[i * R.shape[1] + j])
                        else:
                            new_R_data.append(0.0 if i != j else 1.0)
                R = DenseTensor([r_prev, r_prev], new_R_data)
            else:
                new_R_data = []
                for i in range(r_prev):
                    for j in range(r_prev):
                        new_R_data.append(R.data[i * R.shape[1] + j])
                R = DenseTensor([r_prev, r_prev], new_R_data)
        
        # Заменяем G_k на Q
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
        

        prev_core = cores[k - 1]
        r_prev_prev, n_prev, _ = prev_core.shape
        
        # G_{k-1}[i] = G_{k-1}[i] @ R
        new_prev_data = []
        for i in range(n_prev):
            for p in range(r_prev_prev):
                for q in range(r_prev):
                    val = 0.0
                    for t in range(r_prev):
                        val += prev_core.data[p * n_prev * r_prev + i * r_prev + t] * R.data[t * r_prev + q]
                    new_prev_data.append(val)
        
        cores[k - 1] = DenseTensor([r_prev_prev, n_prev, r_prev], new_prev_data)
    
    return TTTensor(cores)