"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Алгоритм:
    1. Правый проход (полная правая ортогонализация)
    2. Левый проход (SVD-усечение слева направо)
    
    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if tt.order == 1:
        return tt.copy()
    
    # 1. Приводим к право-канонической форме
    tt_right = right_canonicalize(tt, backend)
    cores = [core.copy() for core in tt_right.cores]
    d = tt.order
    
    # Вычисляем норму первого ядра (в право-канонической форме вся норма в G1)
    # Но для надёжности используем полную норму через восстановление?
    # Используем норму первого ядра, т.к. в право-канонической форме
    # ||G1||_F = ||тензор||_F
    norm_first = cores[0].norm()
    
    if norm_first < 1e-30:
        # Нулевой тензор, возвращаем нулевые ядра минимального ранга
        new_cores = []
        for k in range(d):
            r_prev = 1 if k == 0 else 1
            r_next = 1 if k == d - 1 else 1
            core = DenseTensor.zeros([r_prev, tt.shape[k], r_next])
            new_cores.append(core)
        return TTTensor(new_cores)
    
    # Локальный порог усечения
    delta = (eps / math.sqrt(d - 1)) * norm_first if d > 1 else 0.0
    
    r_prev = 1
    
    # 2. Левый проход (SVD-усечение)
    for k in range(d - 1):
        core = cores[k]
        r_prev_core, n_k, r_next_core = core.shape
        
        # Разворачиваем ядро в матрицу размера (r_prev * n_k) x r_next
        G = core.reshape([r_prev * n_k, r_next_core])
        
        # SVD разложение
        U, S, Vt = backend.svd(G, full_matrices=False)
        
        # Вычисляем новый ранг
        new_rank = _compute_rank(S, delta, max_rank)
        
        # Усекаем U до new_rank столбцов
        U_trunc = _truncate_columns(U, new_rank, backend)
        
        # Формируем новое сжатое ядро G_k
        # U_trunc имеет размер (r_prev * n_k, new_rank)
        new_core = U_trunc.reshape([r_prev, n_k, new_rank])
        cores[k] = new_core
        
        # Поглощаем остаток в следующее ядро
        # Остаток: Sigma * Vt^T (размер new_rank x right_dims)
        Sigma = _truncate_vector(S, new_rank, backend)
        Vt_trunc = _truncate_rows(Vt, new_rank, backend)
        
        # Умножаем диагональную матрицу Sigma на Vt_trunc
        # Результат размера (new_rank, Vt_trunc.shape[1])
        absorbed = _multiply_diag_matrix(Sigma, Vt_trunc, new_rank, backend)
        
        # Поглощаем в следующее ядро
        next_core = cores[k + 1]
        r_prev_next, n_next, r_next_next = next_core.shape
        
        # absorbed имеет размер (new_rank, r_next_next * n_next * ...)
        # Нужно умножить absorbed на next_core по первой моде
        # Для каждого индекса i_{k+1}: G_{k+1}[i] = absorbed @ G_{k+1}[i]
        # absorbed имеет размер (new_rank, old_r_prev)
        # next_core имеет размер (old_r_prev, n_next, r_next_next)
        
        # В absorbed хранится Sigma * Vt^T, который имеет размер (new_rank, right_dims)
        # но right_dims = n_{k+1} * n_{k+2} * ... * n_{d-1}
        # Нам нужно перестроить absorbed в матрицу размера (new_rank, old_r_prev)
        # где old_r_prev = r_next_core (старый ранг)
        # А Vt^T имеет размер (old_r_prev, right_dims)
        # absorbed = diag(Sigma) * Vt^T размера (new_rank, right_dims)
        
        # Перестраиваем absorbed в форму для умножения на next_core
        # absorbed должен иметь размер (new_rank, old_r_prev_next)
        # где old_r_prev_next = r_next_core (старый r_prev следующего ядра)
        old_r_prev_next = r_next_core
        
        # Создаём матрицу размера (new_rank, old_r_prev_next)
        # Мы знаем, что absorbed имеет размер (new_rank, right_dims)
        # где right_dims = n_next * r_next_next * ...
        # Нужно свернуть absorbed по размерности right_dims в old_r_prev_next
        # Это не тривиально, проще использовать left_unfolding
        
        # Альтернативный подход: поглощаем через reshape
        # absorbed имеет размер (new_rank, right_dims)
        # Перестраиваем в (new_rank, n_next, old_r_prev_next)
        # Затем для каждого i: G_{k+1}[i] = absorbed[:, i, :] @ G_{k+1}[i]
        
        if k == d - 2:
            # Последнее ядро: absorbed имеет размер (new_rank, n_{d-1})
            # next_core имеет размер (old_r_prev, n_{d-1}, 1)
            # Нужно умножить absorbed на next_core
            absorbed_reshaped = absorbed.reshape([new_rank, n_next])
            new_next_data = []
            for i in range(n_next):
                for p in range(new_rank):
                    val = 0.0
                    for t in range(old_r_prev_next):
                        val += absorbed_reshaped.data[p * n_next + i] * next_core.data[t * n_next + i]
                    new_next_data.append(val)
            new_next_core = DenseTensor([new_rank, n_next, 1], new_next_data)
            cores[k + 1] = new_next_core
        else:
            # Для промежуточных ядер: right_dims = n_{k+1} * r_next_next * ...
            right_dims = absorbed.shape[1]
            # absorbed имеет размер (new_rank, right_dims)
            # Перестраиваем в (new_rank, n_next, old_r_prev_next)
            # где old_r_prev_next = right_dims // n_next
            old_r_prev_next = right_dims // n_next
            
            absorbed_reshaped_data = []
            for p in range(new_rank):
                for i in range(n_next):
                    for t in range(old_r_prev_next):
                        absorbed_reshaped_data.append(
                            absorbed.data[p * right_dims + i * old_r_prev_next + t]
                        )
            absorbed_reshaped = DenseTensor([new_rank, n_next, old_r_prev_next], absorbed_reshaped_data)
            
            # next_core имеет размер (old_r_prev_next, n_next, r_next_next)
            # Для каждого i: G_{k+1}[i] = absorbed[:, i, :] @ G_{k+1}[i]
            new_next_data = []
            for i in range(n_next):
                for p in range(new_rank):
                    for q in range(r_next_next):
                        val = 0.0
                        for t in range(old_r_prev_next):
                            val += absorbed_reshaped.data[p * n_next * old_r_prev_next + i * old_r_prev_next + t] * \
                                   next_core.data[t * n_next * r_next_next + i * r_next_next + q]
                        new_next_data.append(val)
            
            new_next_core = DenseTensor([new_rank, n_next, r_next_next], new_next_data)
            cores[k + 1] = new_next_core
        
        r_prev = new_rank
    
    # Последнее ядро уже обновлено в цикле
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    if S.size == 0:
        return 1
    
    # Числовой ранг: отбрасываем очень маленькие сингулярные значения
    sigma1 = S.data[0] if S.data else 0.0
    numerical_rank = len(S.data)
    
    for i, sigma in enumerate(S.data):
        if sigma <= max(1e-12, 1e-8 * sigma1):
            numerical_rank = i
            break
    
    # Усечение по delta
    truncated_rank = numerical_rank
    
    if delta > 0:
        for r in range(numerical_rank, 0, -1):
            # Сумма квадратов отброшенных (с конца)
            dropped_sq = sum(sigma * sigma for sigma in S.data[r:])
            if dropped_sq <= delta * delta:
                truncated_rank = r
                break
    
    # Применяем ограничение max_rank
    if max_rank is not None:
        truncated_rank = min(truncated_rank, max_rank)
    
    # Гарантируем, что ранг >= 1
    return max(1, truncated_rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

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
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    n = matrix.shape[1]
    data = []
    for i in range(rank):
        for j in range(n):
            data.append(diag_vec.data[i] * matrix.data[i * n + j])
    
    return DenseTensor([rank, n], data)