"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    d = tensor.ndim
    
    # Если тензор порядка 1, просто возвращаем одно ядро
    if d == 1:
        core = tensor.reshape([1, tensor.shape[0], 1])
        return TTTensor([core])
    
    # Вычисляем норму тензора
    norm_tensor = tensor.norm()
    
    # Если норма близка к нулю, возвращаем нулевой TT-тензор
    if norm_tensor < 1e-30:
        cores = []
        for k in range(d):
            r_prev = 1 if k == 0 else 1
            r_next = 1 if k == d - 1 else 1
            core = DenseTensor.zeros([r_prev, tensor.shape[k], r_next])
            cores.append(core)
        return TTTensor(cores)
    
    # Локальный порог усечения
    delta = (eps / math.sqrt(d - 1)) * norm_tensor if d > 1 else 0.0
    
    # Инициализация
    C = tensor.copy()
    cores = []
    r_prev = 1
    
    # Основной цикл по модам
    for k in range(d - 1):
        # Левая развёртка: объединяем первые k+1 мод в строки
        # Остальные моды в столбцы
        left_dims = 1
        for i in range(k + 1):
            left_dims *= C.shape[i]
        right_dims = 1
        for i in range(k + 1, C.ndim):
            right_dims *= C.shape[i]
        
        # Формируем матрицу размера (left_dims, right_dims)
        # Используем unfolding по первой моде (0)
        if C.ndim == 1:
            matrix = C.reshape([left_dims, right_dims])
        else:
            # Разворачиваем по моде 0 (объединяем первые k+1 мод в строки)
            matrix = C.unfolding(0)
        
        # Выполняем SVD
        U, S, Vt = backend.svd(matrix, full_matrices=False)
        
        # Выбираем ранг усечения
        rank = _compute_truncated_rank(S, delta, max_rank)
        
        # Усекаем U до первых rank столбцов
        U_trunc = _truncate_columns(U, rank, backend)
        
        # Формируем ядро G_k: перестраиваем U_trunc в тензор (r_prev, n_k, rank)
        # U_trunc имеет размер (r_prev * n_k, rank)
        G_k_shape = [r_prev, tensor.shape[k], rank]
        G_k_data = []
        for i in range(r_prev * tensor.shape[k]):
            G_k_data.append(U_trunc.data[i * rank:(i + 1) * rank])
        # Создаём DenseTensor из данных
        G_k_flat = []
        for i in range(r_prev):
            for j in range(tensor.shape[k]):
                for l in range(rank):
                    G_k_flat.append(U_trunc.data[(i * tensor.shape[k] + j) * rank + l])
        G_k = DenseTensor(G_k_shape, G_k_flat)
        cores.append(G_k)
        
        # Обновляем C для следующего шага
        # C = Sigma_{1:rank, 1:rank} @ Vt_{1:rank, :}
        # Размер: (rank, right_dims)
        Sigma = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)
        
        # Умножаем диагональную матрицу Sigma на Vt_trunc
        # Результат размера (rank, right_dims)
        new_data = []
        for i in range(rank):
            for j in range(Vt_trunc.shape[1]):
                new_data.append(Sigma.data[i] * Vt_trunc.data[i * Vt_trunc.shape[1] + j])
        
        C = DenseTensor([rank, Vt_trunc.shape[1]], new_data)
        
        # Обновляем r_prev
        r_prev = rank
    
    # Последнее ядро: C имеет размер (r_prev, n_{d-1})
    # Нужно перестроить в (r_prev, n_{d-1}, 1)
    last_shape = [r_prev, tensor.shape[-1], 1]
    last_data = []
    for i in range(r_prev):
        for j in range(tensor.shape[-1]):
            last_data.append(C.data[i * tensor.shape[-1] + j])
    
    G_last = DenseTensor(last_shape, last_data)
    cores.append(G_last)
    
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    # Числовой ранг: отбрасываем очень маленькие сингулярные значения
    sigma1 = S.data[0] if len(S.data) > 0 else 0.0
    numerical_rank = len(S.data)
    
    for i, sigma in enumerate(S.data):
        if sigma <= max(1e-12, 1e-8 * sigma1):
            numerical_rank = i
            break
    
    # Усечение по delta
    # Находим минимальный ранг, при котором сумма квадратов отброшенных
    # сингулярных значений не превышает delta^2
    total_sq = sum(sigma * sigma for sigma in S.data)
    truncated_rank = numerical_rank
    
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
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    n = matrix.shape[1]
    data = []
    for i in range(rank):
        for j in range(n):
            data.append(diag_vec.data[i] * matrix.data[i * n + j])
    
    return DenseTensor([rank, n], data)