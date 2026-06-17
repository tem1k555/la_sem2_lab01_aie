"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

import random
import math
from typing import List, Tuple, Optional, Union

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not cores:
            raise ValueError("TT-tensor must have at least one core")
        
        self.cores = [core.copy() for core in cores]
        self.order = len(cores)
        
        # Проверяем согласованность ядер
        ranks = [1]  # r_0 = 1
        shape = []
        
        for k, core in enumerate(cores):
            if core.ndim != 3:
                raise ValueError(f"Core {k} must be 3D, got {core.ndim}D")
            
            r_prev, n_k, r_next = core.shape
            
            if k == 0 and r_prev != 1:
                raise ValueError(f"First core must have r_0 = 1, got {r_prev}")
            
            if k == len(cores) - 1 and r_next != 1:
                raise ValueError(f"Last core must have r_d = 1, got {r_next}")
            
            if k > 0 and r_prev != ranks[-1]:
                raise ValueError(
                    f"Core {k} has r_prev = {r_prev}, "
                    f"but previous core has r_next = {ranks[-1]}"
                )
            
            ranks.append(r_next)
            shape.append(n_k)
        
        self.shape = tuple(shape)
        self.ranks = tuple(ranks)
        
        # Проверяем, что r_0 = r_d = 1
        if self.ranks[0] != 1 or self.ranks[-1] != 1:
            raise ValueError(f"Ranks must start and end with 1, got {self.ranks}")

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        ranks: tuple[int, ...] | list[int],
        seed: int | None = None
    ) -> TTTensor:
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        if seed is not None:
            random.seed(seed)
        
        shape = validate_shape(shape)
        d = len(shape)
        
        # Если передан только внутренние ранги, добавляем граничные
        if len(ranks) == d - 1:
            ranks = [1] + list(ranks) + [1]
        elif len(ranks) == d + 1:
            ranks = list(ranks)
        else:
            raise ValueError(
                f"ranks must have length {d-1} (internal ranks) "
                f"or {d+1} (full ranks), got {len(ranks)}"
            )
        
        # Проверяем граничные условия
        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError(f"First and last ranks must be 1, got {ranks[0]} and {ranks[-1]}")
        
        cores = []
        for k in range(d):
            r_prev, r_next = ranks[k], ranks[k+1]
            n_k = shape[k]
            
            # Создаём случайное ядро размера (r_prev, n_k, r_next)
            size = r_prev * n_k * r_next
            data = [random.uniform(-1.0, 1.0) for _ in range(size)]
            core = DenseTensor([r_prev, n_k, r_next], data)
            cores.append(core)
        
        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if len(indices) != self.order:
            raise ValueError(
                f"Expected {self.order} indices, got {len(indices)}"
            )
        
        # Проверяем индексы
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= self.shape[i]:
                raise IndexError(
                    f"Index {idx} out of bounds for dimension {i}"
                )
        
        # Вычисляем элемент как произведение матриц G_k[i_k]
        result = None
        for k, core in enumerate(self.cores):
            # Берём срез core[:, i_k, :]
            # Это матрица размера (r_k, r_{k+1})
            r_prev, n_k, r_next = core.shape
            matrix = []
            for i in range(r_prev):
                row = []
                for j in range(r_next):
                    # Индекс в плоском массиве: (i * n_k + idx) * r_next + j
                    flat_idx = (i * n_k + indices[k]) * r_next + j
                    row.append(core.data[flat_idx])
                matrix.append(row)
            
            if result is None:
                result = matrix
            else:
                # Умножаем result на матрицу
                # result имеет размер (1, r_k) или (r_{k-1}, r_k)
                new_result = []
                for i in range(len(result)):
                    row = []
                    for j in range(len(matrix[0])):
                        val = sum(result[i][t] * matrix[t][j] for t in range(len(matrix)))
                        row.append(val)
                    new_result.append(row)
                result = new_result
        
        # В конце должен получиться скаляр (1x1)
        return result[0][0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        size = compute_size(self.shape)
        data = [0.0] * size
        
        def recursive_fill(dim, indices):
            if dim == self.order:
                flat_idx = 0
                for i, idx in enumerate(indices):
                    flat_idx = flat_idx * self.shape[i] + idx
                data[flat_idx] = self.get_element(indices)
                return
            
            for i in range(self.shape[dim]):
                indices.append(i)
                recursive_fill(dim + 1, indices)
                indices.pop()
        
        recursive_fill(0, [])
        return DenseTensor(self.shape, data)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        tt_size = self.total_storage()
        if tt_size == 0:
            return float('inf')
        return full_size / tt_size

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        lines = [
            f"TTTensor(order={self.order}, shape={self.shape})",
            f"  ranks: {self.ranks}",
            f"  core shapes: {self.core_sizes()}",
            f"  total storage: {self.total_storage()} elements",
            f"  compression ratio: {self.compression_ratio():.2f}x",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()