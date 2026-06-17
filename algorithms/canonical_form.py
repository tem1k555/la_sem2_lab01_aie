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
        
        # ★ ИСПРАВЛЕНИЕ: проверяем размеры
        if r_prev * n_k == 0 or r_next == 0:
            continue
        
        # Разворачиваем в матрицу (r_prev * n_k) x r_next
        G = core.reshape([r_prev * n_k, r_next])
        
        # ★ ИСПРАВЛЕНИЕ: если матрица узкая, используем SVD
        if G.shape[0] < G.shape[1]:
            try:
                U, S, Vt = backend.svd(G, full_matrices=False)
                # Берем U как Q
                Q = U
                # R = diag(S) @ Vt
                R_data = []
                for i in range(len(S.data)):
                    for j in range(Vt.shape[1]):
                        R_data.append(S.data[i] * Vt.data[i * Vt.shape[1] + j])
                R = DenseTensor([len(S.data), Vt.shape[1]], R_data)
            except Exception:
                # Если SVD упал, используем QR от транспонированной
                Gt = DenseTensor([r_next, r_prev * n_k], 
                                 [G.data[i * G.shape[1] + j] for j in range(r_prev * n_k) for i in range(r_next)])
                Q_t, R_t = backend.qr(Gt)
                Q_data = []
                for i in range(r_prev * n_k):
                    for j in range(r_next):
                        if j < Q_t.shape[0] and i < Q_t.shape[1]:
                            Q_data.append(Q_t.data[j * Q_t.shape[1] + i])
                        else:
                            Q_data.append(0.0)
                Q = DenseTensor([r_prev * n_k, r_next], Q_data)
                R = R_t
        else:
            # QR-разложение: G = Q @ R
            Q, R = backend.qr(G)
        
        # Заменяем G_k на Q (перестроенный в тензор)
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
        
        # Поглощаем R в следующее ядро
        next_core = cores[k + 1]
        _, n_next, r_next_next = next_core.shape
        
        # ★ ИСПРАВЛЕНИЕ: проверяем размеры R
        if R.shape[0] != r_next:
            if R.shape[0] < r_next:
                # Расширяем R нулями
                new_R_data = []
                for i in range(r_next):
                    for j in range(R.shape[1]):
                        if i < R.shape[0]:
                            new_R_data.append(R.data[i * R.shape[1] + j])
                        else:
                            new_R_data.append(0.0)
                R = DenseTensor([r_next, R.shape[1]], new_R_data)
            else:
                # Усекаем R
                new_R_data = []
                for i in range(r_next):
                    for j in range(R.shape[1]):
                        new_R_data.append(R.data[i * R.shape[1] + j])
                R = DenseTensor([r_next, R.shape[1]], new_R_data)
        
        # G_{k+1}[i] = R @ G_{k+1}[i]
        new_next_data = []
        for i in range(n_next):
            for p in range(r_next):
                for q in range(r_next_next):
                    val = 0.0
                    for t in range(r_next):
                        val += R.data[p * R.shape[1] + t] * next_core.data[t * n_next * r_next_next + i * r_next_next + q]
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
        
        # ★ ИСПРАВЛЕНИЕ: проверяем размеры
        if r_prev == 0 or n_k * r_next == 0:
            continue
        
        # Разворачиваем в матрицу r_prev x (n_k * r_next)
        G = core.reshape([r_prev, n_k * r_next])
        
        # ★ ИСПРАВЛЕНИЕ: если матрица узкая, используем SVD
        if r_prev > n_k * r_next:
            # Транспонируем матрицу
            Gt_data = []
            for j in range(n_k * r_next):
                for i in range(r_prev):
                    Gt_data.append(G.data[i * (n_k * r_next) + j])
            Gt = DenseTensor([n_k * r_next, r_prev], Gt_data)
            
            try:
                Q_t, R_t = backend.qr(Gt)
            except Exception:
                U, S, Vt = backend.svd(Gt, full_matrices=False)
                Q_t = U
                R_t_data = []
                for i in range(U.shape[1]):
                    for j in range(Gt.shape[1]):
                        R_t_data.append(U.data[i * U.shape[1] + j] * S.data[i])
                R_t = DenseTensor([U.shape[1], Gt.shape[1]], R_t_data)
            
            # Q = Q_t^T, R = R_t^T
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
        else:
            # Выполняем QR от транспонированной: G^T = Q_t @ R_t
            Gt_data = []
            for j in range(r_prev):
                for i in range(n_k * r_next):
                    Gt_data.append(G.data[i * r_prev + j])
            Gt = DenseTensor([n_k * r_next, r_prev], Gt_data)
            
            try:
                Q_t, R_t = backend.qr(Gt)
            except Exception:
                # Если QR упал, используем SVD
                U, S, Vt = backend.svd(Gt, full_matrices=False)
                Q_t = U
                R_t_data = []
                for i in range(U.shape[1]):
                    for j in range(Gt.shape[1]):
                        R_t_data.append(U.data[i * U.shape[1] + j] * S.data[i])
                R_t = DenseTensor([U.shape[1], Gt.shape[1]], R_t_data)
            
            # Q = Q_t^T, R = R_t^T
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
        
        # Заменяем G_k на Q (перестроенный в тензор)
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
        
        # Поглощаем R в предыдущее ядро
        prev_core = cores[k - 1]
        r_prev_prev, n_prev, _ = prev_core.shape
        
        # ★ ИСПРАВЛЕНИЕ: проверяем размеры R
        if R.shape[0] != r_prev or R.shape[1] != r_prev:
            if R.shape[0] < r_prev or R.shape[1] < r_prev:
                new_R_data = []
                for i in range(r_prev):
                    for j in range(r_prev):
                        if i < R.shape[0] and j < R.shape[1]:
                            new_R_data.append(R.data[i * R.shape[1] + j])
                        else:
                            new_R_data.append(0.0)
                R = DenseTensor([r_prev, r_prev], new_R_data)
            else:
                new_R_data = []
                for i in range(r_prev):
                    for j in range(r_prev):
                        new_R_data.append(R.data[i * R.shape[1] + j])
                R = DenseTensor([r_prev, r_prev], new_R_data)
        
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