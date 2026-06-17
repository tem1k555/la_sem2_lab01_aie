"""Функции для работы с тензорами в стандартной плотной форме."""

from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
    check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)
        
        if data is not None:
            if len(data) != self.size:
                raise ValueError(
                    f"Data length {len(data)} does not match tensor size {self.size}"
                )
            self.data = [float(x) for x in data]
        else:
            self.data = [float(fill)] * self.size

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        if seed is not None:
            random.seed(seed)
        
        size = compute_size(validate_shape(shape))
        if integer:
            data = [random.randint(low, high) for _ in range(size)]
        else:
            data = [random.uniform(low, high) for _ in range(size)]
        
        return DenseTensor(shape, data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        def get_shape(lst):
            if not isinstance(lst, list):
                return ()
            if not lst:
                return (0,)
            first_shape = get_shape(lst[0])
            return (len(lst),) + first_shape
        
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(float(item))
            return result
        
        shape = get_shape(nested)
        # Проверяем, что все подсписки имеют одинаковую форму
        def check_shape(lst, expected_shape, depth):
            if depth >= len(expected_shape):
                return
            if not isinstance(lst, list):
                raise ValueError("Inconsistent nested list structure")
            if len(lst) != expected_shape[depth]:
                raise ValueError("Inconsistent nested list structure")
            for item in lst:
                check_shape(item, expected_shape, depth + 1)
        
        check_shape(nested, shape, 0)
        
        data = flatten(nested)
        return DenseTensor(shape, data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int):
            if multi_index < 0 or multi_index >= self.size:
                raise IndexError(f"Index {multi_index} out of bounds")
            return flat_to_multi_index(multi_index, self.shape)
        
        if len(multi_index) != self.ndim:
            raise IndexError(
                f"Expected {self.ndim} indices, got {len(multi_index)}"
            )
        
        for i, idx in enumerate(multi_index):
            if idx < 0 or idx >= self.shape[i]:
                raise IndexError(
                    f"Index {idx} out of bounds for dimension {i} (size {self.shape[i]})"
                )
        
        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        idx_tuple = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        return self.data[flat_idx]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        idx_tuple = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        self.data[flat_idx] = float(value)

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        new_shape = validate_shape(new_shape)
        if compute_size(new_shape) != self.size:
            raise ValueError(
                f"Cannot reshape {self.shape} to {new_shape}: sizes differ"
            )
        return DenseTensor(new_shape, self.data.copy())

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Переставляем оси так, чтобы mode стала первой
        order = [mode] + [i for i in range(self.ndim) if i != mode]
        
        # Получаем данные в новом порядке
        # Для этого нужно правильно переставить элементы
        # Создаём новый тензор с переставленными осями
        transposed_data = [0.0] * self.size
        
        # Для каждого мультииндекса в исходном тензоре
        # находим соответствующий индекс в переставленном
        def get_indices_recursive(dim, current_indices):
            if dim == self.ndim:
                # Исходный мультииндекс
                old_idx = tuple(current_indices)
                flat_old = multi_index_to_flat(old_idx, self.strides)
                
                # Новый мультииндекс (переставленный)
                new_indices = [old_idx[i] for i in order]
                new_flat = multi_index_to_flat(
                    tuple(new_indices),
                    compute_strides(tuple(self.shape[i] for i in order))
                )
                
                transposed_data[new_flat] = self.data[flat_old]
                return
            
            for i in range(self.shape[dim]):
                current_indices.append(i)
                get_indices_recursive(dim + 1, current_indices)
                current_indices.pop()
        
        get_indices_recursive(0, [])
        
        # Новая форма: (n_mode, n_1 * ... * n_{mode-1} * n_{mode+1} * ...)
        row_dim = self.shape[mode]
        col_dim = self.size // row_dim
        
        return DenseTensor([row_dim, col_dim], transposed_data)

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(f"Invalid k: {k}, must be in [0, {self.ndim - 2}]")
        
        # Левая развертка: объединяем первые k+1 мод в строки
        # Остальные моды объединяем в столбцы
        left_dims = compute_size(self.shape[:k+1])
        right_dims = compute_size(self.shape[k+1:])
        
        # Создаём матрицу размера left_dims x right_dims
        data = [0.0] * (left_dims * right_dims)
        
        def fill_recursive(dim, current_indices):
            if dim == self.ndim:
                # Получаем плоский индекс для исходного тензора
                old_flat = multi_index_to_flat(tuple(current_indices), self.strides)
                value = self.data[old_flat]
                
                # Вычисляем новый индекс в матрице
                # Строка: мультииндекс из первых k+1 мод -> плоский индекс
                left_idx = 0
                for i in range(k + 1):
                    left_idx = left_idx * self.shape[i] + current_indices[i]
                
                # Столбец: мультииндекс из оставшихся мод -> плоский индекс
                right_idx = 0
                for i in range(k + 1, self.ndim):
                    right_idx = right_idx * self.shape[i] + current_indices[i]
                
                new_flat = left_idx * right_dims + right_idx
                data[new_flat] = value
                return
            
            for i in range(self.shape[dim]):
                current_indices.append(i)
                fill_recursive(dim + 1, current_indices)
                current_indices.pop()
        
        fill_recursive(0, [])
        
        return DenseTensor([left_dims, right_dims], data)

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, self.data.copy())

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        result_data = [a + b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, result_data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        result_data = [a - b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, result_data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        scalar = float(scalar)
        result_data = [x * scalar for x in self.data]
        return DenseTensor(self.shape, result_data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self.__mul__(-1.0)

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)

        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if self.shape != other.shape:
            return False
        
        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        
        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        def build_nested(indices, dim):
            if dim == self.ndim:
                return self[tuple(indices)]
            
            result = []
            for i in range(self.shape[dim]):
                indices.append(i)
                result.append(build_nested(indices, dim + 1))
                indices.pop()
            return result
        
        return build_nested([], 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.to_nested_list()})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()