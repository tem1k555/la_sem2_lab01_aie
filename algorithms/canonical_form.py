"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Лево-каноническая форма: все ядра кроме последнего являются
    лево-ортогональными, т.е. для каждого ядра G_k (k < d):
        sum_{i_k} G_k[i_k]^T @ G_k[i_k] = I

    Алгоритм: проход слева направо (k = 0, 1, ..., d-2)
    1. Разворачиваем G_k в матрицу размера (r_{k-1} * n_k) x r_k
    2. Выполняем QR-разложение: G = QR
    3. Заменяем G_k на Q (перестроенный в тензор (r_{k-1}, n_k, r_k))
    4. Поглощаем R в следующее ядро G_{k+1}

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    if tt.order == 1:
        return tt.copy()
    
    cores = [core.copy() for core in tt.cores]
    d = tt.order
    
    for k in range(d - 1):
        core = cores[k]
        r_prev, n_k, r_next = core.shape
        
        # 1. Разворачиваем ядро в матрицу размера (r_prev * n_k) x r_next
        G = core.reshape([r_prev * n_k, r_next])
        
        # 2. QR-разложение
        Q, R = backend.qr(G)
        
        # 3. Заменяем G_k на Q (перестроенный в тензор)
        # Q имеет размер (r_prev * n_k, r_next)
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
        
        # 4. Поглощаем R в следующее ядро
        # R имеет размер (r_next, r_next)
        next_core = cores[k + 1]
        r_prev_next, n_next, r_next_next = next_core.shape
        
        # R имеет размер (r_next, r_next)
        # Нужно умножить R на G_{k+1} по первой моде (r_prev)
        # Для каждого индекса i_{k+1}: G_{k+1}[i_{k+1}] = R @ G_{k+1}[i_{k+1}]
        new_next_data = []
        for i in range(n_next):
            # Матрица G_{k+1}[i] размера (r_next, r_next_next)
            # Берём срез core[:, i, :]
            for p in range(r_next):
                for q in range(r_next_next):
                    val = 0.0
                    for t in range(r_next):
                        val += R.data[p * r_next + t] * next_core.data[t * n_next * r_next_next + i * r_next_next + q]
                    new_next_data.append(val)
        
        new_next_core = DenseTensor([r_next, n_next, r_next_next], new_next_data)
        cores[k + 1] = new_next_core
    
    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Право-каноническая форма: все ядра кроме первого являются
    право-ортогональными, т.е. для каждого ядра G_k (k > 0):
        sum_{i_k} G_k[i_k] @ G_k[i_k]^T = I

    Алгоритм: проход справа налево (k = d-1, d-2, ..., 1)
    1. Разворачиваем G_k в матрицу размера r_{k-1} x (n_k * r_k)
    2. Выполняем RQ-разложение (или QR от транспонированной): G = RQ
    3. Заменяем G_k на Q (перестроенный в тензор (r_{k-1}, n_k, r_k))
    4. Поглощаем R в предыдущее ядро G_{k-1}

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    if tt.order == 1:
        return tt.copy()
    
    cores = [core.copy() for core in tt.cores]
    d = tt.order
    
    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_prev, n_k, r_next = core.shape
        
        # 1. Разворачиваем ядро в матрицу размера r_prev x (n_k * r_next)
        G = core.reshape([r_prev, n_k * r_next])
        
        # 2. Выполняем QR разложение от транспонированной
        # G^T имеет размер (n_k * r_next) x r_prev
        Gt = G.reshape([n_k * r_next, r_prev])  # Транспонируем
        # В реальности нужно транспонировать данные
        Gt_data = []
        for j in range(n_k * r_next):
            for i in range(r_prev):
                Gt_data.append(G.data[i * (n_k * r_next) + j])
        Gt = DenseTensor([n_k * r_next, r_prev], Gt_data)
        
        Q_t, R_t = backend.qr(Gt)
        
        # Q_t имеет размер (n_k * r_next, r_prev)
        # R_t имеет размер (r_prev, r_prev)
        
        # Вычисляем Q = Q_t^T, R = R_t^T
        # Q должен иметь размер (r_prev, n_k * r_next)
        Q_data = []
        for i in range(r_prev):
            for j in range(n_k * r_next):
                Q_data.append(Q_t.data[j * r_prev + i])
        Q = DenseTensor([r_prev, n_k * r_next], Q_data)
        
        # R должен иметь размер (r_prev, r_prev)
        R_data = []
        for i in range(r_prev):
            for j in range(r_prev):
                R_data.append(R_t.data[j * r_prev + i])
        R = DenseTensor([r_prev, r_prev], R_data)
        
        # 3. Заменяем G_k на Q (перестроенный в тензор)
        Q_reshaped = Q.reshape([r_prev, n_k, r_next])
        cores[k] = Q_reshaped
        
        # 4. Поглощаем R в предыдущее ядро
        # R имеет размер (r_prev, r_prev)
        prev_core = cores[k - 1]
        r_prev_prev, n_prev, r_prev_next = prev_core.shape
        
        # Нужно умножить G_{k-1} на R справа: G_{k-1}[i] = G_{k-1}[i] @ R
        new_prev_data = []
        for i in range(n_prev):
            # Матрица G_{k-1}[i] размера (r_prev_prev, r_prev)
            for p in range(r_prev_prev):
                for q in range(r_prev):
                    val = 0.0
                    for t in range(r_prev):
                        val += prev_core.data[p * n_prev * r_prev + i * r_prev + t] * R.data[t * r_prev + q]
                    new_prev_data.append(val)
        
        # Обновляем размер: r_prev_next должно стать r_prev
        new_prev_core = DenseTensor([r_prev_prev, n_prev, r_prev], new_prev_data)
        cores[k - 1] = new_prev_core
    
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.size == 0:
        return 0
    
    sigma_max = S.data[0] if S.data else 0.0
    threshold = max(abs_tol, rel_tol * sigma_max)
    
    rank = 0
    for sigma in S.data:
        if sigma > threshold:
            rank += 1
        else:
            break
    
    return rank


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
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
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
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
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if rank >= vector.size:
        return vector.copy()
    
    return DenseTensor([rank], vector.data[:rank])


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    n = matrix.shape[1]
    data = []
    for i in range(rank):
        for j in range(n):
            data.append(diag_vec.data[i] * matrix.data[i * n + j])
    
    return DenseTensor([rank, n], data)


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    m, n = matrix.shape
    if n != diag_vec.size:
        raise ValueError(f"Matrix columns {n} != diag_vec size {diag_vec.size}")
    
    data = []
    for i in range(m):
        for j in range(n):
            data.append(matrix.data[i * n + j] * diag_vec.data[j])
    
    return DenseTensor([m, n], data)