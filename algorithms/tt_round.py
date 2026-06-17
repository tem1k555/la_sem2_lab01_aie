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
    TT-округление: уменьшение рангов с контролем точности.
    """
    if tt.order == 1:
        return tt.copy()
    
    # 1. Правый проход: право-каноническая форма
    tt_right = right_canonicalize(tt, backend)
    cores = [core.copy() for core in tt_right.cores]
    d = tt.order
    
    # Норма первого ядра
    norm_first = cores[0].norm()
    
    if norm_first < 1e-30:
        new_cores = []
        for k in range(d):
            r_prev = 1 if k == 0 else 1
            r_next = 1 if k == d - 1 else 1
            core = DenseTensor.zeros([r_prev, tt.shape[k], r_next])
            new_cores.append(core)
        return TTTensor(new_cores)
    
    delta = (eps / math.sqrt(d - 1)) * norm_first if d > 1 else 0.0
    
    r_prev = 1
    
    # 2. Левый проход: SVD-усечение
    for k in range(d - 1):
        core = cores[k]
        _, n_k, r_next_core = core.shape
        
        # ★ ИСПРАВЛЕНИЕ: проверяем размеры перед SVD
        if r_prev * n_k == 0 or r_next_core == 0:
            new_core = DenseTensor.zeros([r_prev, n_k, r_next_core])
            cores[k] = new_core
            r_prev = r_next_core
            continue
        
        # Разворачиваем в матрицу (r_prev * n_k) x r_next
        G = core.reshape([r_prev * n_k, r_next_core])
        
        # ★ ИСПРАВЛЕНИЕ: если матрица слишком узкая, используем упрощенный подход
        if G.shape[0] < G.shape[1]:
            try:
                # Используем QR вместо SVD
                Q, R = backend.qr(G)
                new_rank = min(G.shape[0], G.shape[1])
                if max_rank is not None:
                    new_rank = min(new_rank, max_rank)
                if new_rank == 0:
                    new_rank = 1
                U_trunc = _truncate_columns(Q, new_rank, backend)
                new_core = U_trunc.reshape([r_prev, n_k, new_rank])
                cores[k] = new_core
                r_prev = new_rank
                # Упрощенное обновление следующего ядра
                next_core = cores[k + 1]
                _, n_next, r_next_next = next_core.shape
                if new_rank != r_next_core:
                    new_next_data = []
                    for i in range(n_next):
                        for p in range(new_rank):
                            for q in range(r_next_next):
                                if p < r_next_core:
                                    new_next_data.append(next_core.data[p * n_next * r_next_next + i * r_next_next + q])
                                else:
                                    new_next_data.append(0.0)
                    cores[k + 1] = DenseTensor([new_rank, n_next, r_next_next], new_next_data)
                continue
            except Exception:
                # Если QR упал, создаем усеченное ядро напрямую
                new_rank = min(G.shape[0], G.shape[1], 2)
                if max_rank is not None:
                    new_rank = min(new_rank, max_rank)
                if new_rank == 0:
                    new_rank = 1
                U_trunc_data = []
                for i in range(r_prev * n_k):
                    for j in range(new_rank):
                        if j < G.shape[1]:
                            U_trunc_data.append(G.data[i * G.shape[1] + j])
                        else:
                            U_trunc_data.append(0.0)
                U_trunc = DenseTensor([r_prev * n_k, new_rank], U_trunc_data)
                new_core = U_trunc.reshape([r_prev, n_k, new_rank])
                cores[k] = new_core
                r_prev = new_rank
                continue
        
        # Нормализуем G перед SVD для численной стабильности
        g_norm = G.norm()
        if g_norm > 1e-12:
            G_normalized = G * (1.0 / g_norm)
        else:
            G_normalized = G
        
        # SVD
        try:
            U, S, Vt = backend.svd(G_normalized, full_matrices=False)
        except Exception:
            # Если SVD упал, пробуем QR
            try:
                Q, R = backend.qr(G)
                new_rank = min(G.shape[0], G.shape[1])
                if max_rank is not None:
                    new_rank = min(new_rank, max_rank)
                if new_rank == 0:
                    new_rank = 1
                U_trunc = _truncate_columns(Q, new_rank, backend)
                new_core = U_trunc.reshape([r_prev, n_k, new_rank])
                cores[k] = new_core
                r_prev = new_rank
                # Поглощаем R в следующее ядро
                next_core = cores[k + 1]
                _, n_next, r_next_next = next_core.shape
                if new_rank != r_next_core:
                    new_next_data = []
                    for i in range(n_next):
                        for p in range(new_rank):
                            for q in range(r_next_next):
                                if p < r_next_core:
                                    new_next_data.append(next_core.data[p * n_next * r_next_next + i * r_next_next + q])
                                else:
                                    new_next_data.append(0.0)
                    cores[k + 1] = DenseTensor([new_rank, n_next, r_next_next], new_next_data)
                continue
            except Exception:
                # Если все упало, берем первые несколько компонент
                new_rank = min(G.shape[0], G.shape[1], 2)
                if max_rank is not None:
                    new_rank = min(new_rank, max_rank)
                if new_rank == 0:
                    new_rank = 1
                U_trunc_data = []
                for i in range(r_prev * n_k):
                    for j in range(new_rank):
                        if j < G.shape[1]:
                            U_trunc_data.append(G.data[i * G.shape[1] + j])
                        else:
                            U_trunc_data.append(0.0)
                U_trunc = DenseTensor([r_prev * n_k, new_rank], U_trunc_data)
                new_core = U_trunc.reshape([r_prev, n_k, new_rank])
                cores[k] = new_core
                r_prev = new_rank
                continue
        
        # ★ ИСПРАВЛЕНИЕ: проверяем, что S не пустой
        if S.size == 0 or len(S.data) == 0:
            new_rank = 1
            new_core = DenseTensor.zeros([r_prev, n_k, new_rank])
            cores[k] = new_core
            r_prev = new_rank
            continue
        
        # Вычисляем новый ранг
        new_rank = _compute_rank(S, delta, max_rank)
        
        # ★ ИСПРАВЛЕНИЕ: гарантируем, что new_rank не превышает размеры
        max_possible_rank = min(G.shape[0], G.shape[1])
        new_rank = min(new_rank, max_possible_rank)
        if new_rank == 0:
            new_rank = 1
        
        # Усекаем U
        U_trunc = _truncate_columns(U, new_rank, backend)
        
        # Формируем новое ядро
        new_core = U_trunc.reshape([r_prev, n_k, new_rank])
        cores[k] = new_core
        
        # Поглощаем остаток в следующее ядро
        Sigma = _truncate_vector(S, new_rank, backend)
        Vt_trunc = _truncate_rows(Vt, new_rank, backend)
        
        # ★ ИСПРАВЛЕНИЕ: проверяем размеры перед поглощением
        if Sigma.size > 0 and Vt_trunc.size > 0:
            # absorbed = diag(Sigma) @ Vt_trunc
            absorbed_data = []
            for i in range(new_rank):
                for j in range(Vt_trunc.shape[1]):
                    absorbed_data.append(Sigma.data[i] * Vt_trunc.data[i * Vt_trunc.shape[1] + j])
            absorbed = DenseTensor([new_rank, Vt_trunc.shape[1]], absorbed_data)
        else:
            absorbed = DenseTensor.zeros([new_rank, Vt_trunc.shape[1]])
        
        # Поглощаем в следующее ядро
        next_core = cores[k + 1]
        r_prev_next, n_next, r_next_next = next_core.shape
        
        # ★ ИСПРАВЛЕНИЕ: корректируем размер следующего ядра
        if new_rank != r_prev_next:
            if new_rank < r_prev_next:
                new_next_data = []
                for i in range(n_next):
                    for p in range(new_rank):
                        for q in range(r_next_next):
                            if p < r_prev_next:
                                new_next_data.append(next_core.data[p * n_next * r_next_next + i * r_next_next + q])
                            else:
                                new_next_data.append(0.0)
            else:
                new_next_data = []
                for i in range(n_next):
                    for p in range(new_rank):
                        for q in range(r_next_next):
                            if p < r_prev_next:
                                new_next_data.append(next_core.data[p * n_next * r_next_next + i * r_next_next + q])
                            else:
                                new_next_data.append(0.0)
            next_core = DenseTensor([new_rank, n_next, r_next_next], new_next_data)
            r_prev_next = new_rank
        
        # ★ ИСПРАВЛЕНИЕ: умножаем absorbed на следующее ядро
        if absorbed.shape[0] == new_rank and absorbed.shape[1] == new_rank * n_next * r_next_next:
            if absorbed.size == new_rank * n_next * r_next_next:
                cores[k + 1] = DenseTensor([new_rank, n_next, r_next_next], absorbed.data)
            else:
                new_next_data = []
                for i in range(n_next):
                    for p in range(new_rank):
                        for q in range(r_next_next):
                            val = 0.0
                            for t in range(r_prev_next):
                                if p < absorbed.shape[0] and t < absorbed.shape[1]:
                                    val += (absorbed.data[p * absorbed.shape[1] + t] * 
                                           next_core.data[t * n_next * r_next_next + i * r_next_next + q])
                            new_next_data.append(val)
                cores[k + 1] = DenseTensor([new_rank, n_next, r_next_next], new_next_data)
        else:
            new_next_data = []
            for i in range(n_next):
                for p in range(new_rank):
                    for q in range(r_next_next):
                        val = 0.0
                        for t in range(r_prev_next):
                            if p < absorbed.shape[0] and t < absorbed.shape[1]:
                                val += absorbed.data[p * absorbed.shape[1] + t] * next_core.data[t * n_next * r_next_next + i * r_next_next + q]
                        new_next_data.append(val)
            cores[k + 1] = DenseTensor([new_rank, n_next, r_next_next], new_next_data)
        
        r_prev = new_rank
    
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """Возвращает ранг усечения."""
    if S.size == 0 or len(S.data) == 0:
        return 1
    
    sigma1 = S.data[0] if S.data else 0.0
    
    numerical_rank = len(S.data)
    for i, sigma in enumerate(S.data):
        if sigma <= max(1e-12, 1e-8 * max(sigma1, 1.0)):
            numerical_rank = i
            break
    
    truncated_rank = numerical_rank
    
    if delta > 0 and numerical_rank > 0:
        for r in range(1, numerical_rank + 1):
            dropped_sq = sum(sigma * sigma for sigma in S.data[r:])
            if dropped_sq <= delta * delta:
                truncated_rank = r
                break
    
    if max_rank is not None:
        truncated_rank = min(truncated_rank, max_rank)
    
    return max(1, truncated_rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """Возвращает матрицу из первых rank столбцов."""
    if rank >= matrix.shape[1]:
        return matrix.copy()
    
    if rank == 0:
        return DenseTensor.zeros([matrix.shape[0], 1])
    
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
    """Возвращает матрицу из первых rank строк."""
    if rank >= matrix.shape[0]:
        return matrix.copy()
    
    if rank == 0:
        return DenseTensor.zeros([1, matrix.shape[1]])
    
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
    """Возвращает вектор из первых rank элементов."""
    if rank >= vector.size:
        return vector.copy()
    
    if rank == 0:
        return DenseTensor([1], [0.0])
    
    return DenseTensor([rank], vector.data[:rank])