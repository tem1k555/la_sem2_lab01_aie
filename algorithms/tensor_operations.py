"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Формула:
    G_1^C[i_1] = [G_1^A[i_1], G_1^B[i_1]]
    G_k^C[i_k] = [[G_k^A[i_k], 0], [0, G_k^B[i_k]]] для k = 2, ..., d-1
    G_d^C[i_d] = [G_d^A[i_d], G_d^B[i_d]]^T

    TT-ранги результата: r_k^C = r_k^A + r_k^B

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    d = tt1.order
    cores1 = tt1.cores
    cores2 = tt2.cores
    
    new_cores = []
    
    for k in range(d):
        core1 = cores1[k]
        core2 = cores2[k]
        r1_prev, n_k, r1_next = core1.shape
        r2_prev, n_k2, r2_next = core2.shape
        
        if k == 0:
            # Первое ядро: [G_1^A[i_1], G_1^B[i_1]]
            # Размер: (1, n_k, r1_next + r2_next)
            new_r_prev = 1
            new_r_next = r1_next + r2_next
            
            new_data = []
            for i in range(n_k):
                # Строка G_1^A[i_1] размера (1, r1_next)
                for j in range(r1_next):
                    new_data.append(core1.data[0 * n_k * r1_next + i * r1_next + j])
                # Строка G_1^B[i_1] размера (1, r2_next)
                for j in range(r2_next):
                    new_data.append(core2.data[0 * n_k * r2_next + i * r2_next + j])
            
            new_core = DenseTensor([new_r_prev, n_k, new_r_next], new_data)
            
        elif k == d - 1:
            # Последнее ядро: [G_d^A[i_d], G_d^B[i_d]]^T
            # Размер: (r1_prev + r2_prev, n_k, 1)
            new_r_prev = r1_prev + r2_prev
            new_r_next = 1
            
            new_data = []
            for i in range(n_k):
                # Строки G_d^A[i_d] размера (r1_prev, 1)
                for j in range(r1_prev):
                    new_data.append(core1.data[j * n_k * 1 + i * 1 + 0])
                # Строки G_d^B[i_d] размера (r2_prev, 1)
                for j in range(r2_prev):
                    new_data.append(core2.data[j * n_k * 1 + i * 1 + 0])
            
            new_core = DenseTensor([new_r_prev, n_k, new_r_next], new_data)
            
        else:
            # Промежуточные ядра: блочно-диагональные
            # [[G_k^A[i_k], 0], [0, G_k^B[i_k]]]
            # Размер: (r1_prev + r2_prev, n_k, r1_next + r2_next)
            new_r_prev = r1_prev + r2_prev
            new_r_next = r1_next + r2_next
            
            new_data = []
            for i in range(n_k):
                for p in range(new_r_prev):
                    for q in range(new_r_next):
                        if p < r1_prev and q < r1_next:
                            # Верхний левый блок: G_k^A
                            val = core1.data[p * n_k * r1_next + i * r1_next + q]
                        elif p >= r1_prev and q >= r1_next:
                            # Нижний правый блок: G_k^B
                            p2 = p - r1_prev
                            q2 = q - r1_next
                            val = core2.data[p2 * n_k * r2_next + i * r2_next + q2]
                        else:
                            val = 0.0
                        new_data.append(val)
            
            new_core = DenseTensor([new_r_prev, n_k, new_r_next], new_data)
        
        new_cores.append(new_core)
    
    return TTTensor(new_cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    if alpha == 1.0:
        return tt.copy()
    
    cores = [core.copy() for core in tt.cores]
    
    # Умножаем первое ядро на alpha
    core0 = cores[0]
    r_prev, n_0, r_next = core0.shape
    new_data = [alpha * x for x in core0.data]
    cores[0] = DenseTensor([r_prev, n_0, r_next], new_data)
    
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Формула: G_k^C[i_k] = G_k^A[i_k] ⊗ G_k^B[i_k] (кронекерово произведение)

    TT-ранги результата: r_k^C = r_k^A * r_k^B

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    d = tt1.order
    cores1 = tt1.cores
    cores2 = tt2.cores
    
    new_cores = []
    
    for k in range(d):
        core1 = cores1[k]
        core2 = cores2[k]
        r1_prev, n_k, r1_next = core1.shape
        r2_prev, n_k2, r2_next = core2.shape
        
        # Кронекерово произведение: (r1_prev * r2_prev, n_k, r1_next * r2_next)
        new_r_prev = r1_prev * r2_prev
        new_r_next = r1_next * r2_next
        
        new_data = []
        for i in range(n_k):
            # Для каждого i, матрица G_k^C[i] = G_k^A[i] ⊗ G_k^B[i]
            # G_k^A[i] размера (r1_prev, r1_next)
            # G_k^B[i] размера (r2_prev, r2_next)
            for p1 in range(r1_prev):
                for p2 in range(r2_prev):
                    for q1 in range(r1_next):
                        for q2 in range(r2_next):
                            val = (core1.data[p1 * n_k * r1_next + i * r1_next + q1] *
                                   core2.data[p2 * n_k * r2_next + i * r2_next + q2])
                            new_data.append(val)
        
        new_core = DenseTensor([new_r_prev, n_k, new_r_next], new_data)
        new_cores.append(new_core)
    
    return TTTensor(new_cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Алгоритм:
    Z_1 = sum_{i_1} G_1^A[i_1]^T @ G_1^B[i_1]
    Z_k = sum_{i_k} G_k^A[i_k]^T @ Z_{k-1} @ G_k^B[i_k]
    result = Z_d (скаляр)

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    d = tt1.order
    cores1 = tt1.cores
    cores2 = tt2.cores
    
    # Z_1 = sum_{i_1} G_1^A[i_1]^T @ G_1^B[i_1]
    # G_1^A[i_1] размера (1, r1_1)
    # G_1^B[i_1] размера (1, r2_1)
    # Результат размера (r1_1, r2_1)
    core1_0 = cores1[0]
    core2_0 = cores2[0]
    r1_1 = core1_0.shape[2]  # r1_next
    r2_1 = core2_0.shape[2]  # r2_next
    n_0 = core1_0.shape[1]
    
    Z_data = []
    for i in range(r1_1):
        row = []
        for j in range(r2_1):
            val = 0.0
            for idx in range(n_0):
                val += (core1_0.data[0 * n_0 * r1_1 + idx * r1_1 + i] *
                        core2_0.data[0 * n_0 * r2_1 + idx * r2_1 + j])
            row.append(val)
        Z_data.extend(row)
    
    Z = DenseTensor([r1_1, r2_1], Z_data)
    
    # Для k = 2, ..., d
    for k in range(1, d):
        core1 = cores1[k]
        core2 = cores2[k]
        r1_prev, n_k, r1_next = core1.shape
        r2_prev, n_k2, r2_next = core2.shape
        
        # Z_k = sum_{i_k} G_k^A[i_k]^T @ Z_{k-1} @ G_k^B[i_k]
        # G_k^A[i_k] размера (r1_prev, r1_next)
        # G_k^B[i_k] размера (r2_prev, r2_next)
        # Z_{k-1} размера (r1_prev, r2_prev)
        # Результат размера (r1_next, r2_next)
        
        new_Z_data = []
        for i in range(r1_next):
            for j in range(r2_next):
                val = 0.0
                for idx in range(n_k):
                    # Суммируем по idx (i_k)
                    for p in range(r1_prev):
                        for q in range(r2_prev):
                            val += (core1.data[p * n_k * r1_next + idx * r1_next + i] *
                                    Z.data[p * r2_prev + q] *
                                    core2.data[q * n_k * r2_next + idx * r2_next + j])
                new_Z_data.append(val)
        
        Z = DenseTensor([r1_next, r2_next], new_Z_data)
    
    # В конце Z должен быть размера (1, 1) - скаляр
    return Z.data[0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    return math.sqrt(tt_dot(tt, tt, backend))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    ||tt1 - tt2||_F^2 = ||tt1||_F^2 + ||tt2||_F^2 - 2 * <tt1, tt2>

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"Shapes must match: {tt1.shape} vs {tt2.shape}")
    
    norm1_sq = tt_norm(tt1, backend) ** 2
    norm2_sq = tt_norm(tt2, backend) ** 2
    dot = tt_dot(tt1, tt2, backend)
    
    diff_sq = norm1_sq + norm2_sq - 2 * dot
    
    # Избегаем отрицательных значений из-за ошибок округления
    if diff_sq < 0 and diff_sq > -1e-10:
        diff_sq = 0.0
    
    return math.sqrt(diff_sq)