"""
Плотный тензор произвольного порядка.
"""

from __future__ import annotations

import random
import math
from typing import Optional, Union

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
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: Optional[list[float]] = None,
        fill: float = 0.0
    ) -> None:
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
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: Optional[int] = None
    ) -> DenseTensor:
        if seed is not None:
            random.seed(seed)
        
        size = compute_size(validate_shape(shape))
        if integer:
            data = [float(random.randint(low, high)) for _ in range(size)]
        else:
            data = [random.uniform(low, high) for _ in range(size)]
        
        return DenseTensor(shape, data=data)

    @staticmethod
    def from_nested_list(nested: list | tuple) -> DenseTensor:
        def get_shape(lst):
            if not isinstance(lst, (list, tuple)):
                return ()
            if len(lst) == 0:
                return (0,)
            return (len(lst),) + get_shape(lst[0])
        
        def flatten(lst):
            if not isinstance(lst, (list, tuple)):
                return [float(lst)]
            result = []
            for item in lst:
                result.extend(flatten(item))
            return result
        
        shape = get_shape(nested)
        data = flatten(nested)
        return DenseTensor(shape, data=data)

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        if isinstance(multi_index, int):
            if multi_index < 0 or multi_index >= self.size:
                raise IndexError(f"Index {multi_index} out of bounds")
            return flat_to_multi_index(multi_index, self.shape)
        
        if len(multi_index) != self.ndim:
            raise IndexError(
                f"Expected {self.ndim} indices, got {len(multi_index)}"
            )
        
        for dim_idx, idx in enumerate(multi_index):
            if idx < 0 or idx >= self.shape[dim_idx]:
                raise IndexError(
                    f"Index {idx} out of bounds for dimension {dim_idx} "
                    f"(size {self.shape[dim_idx]})"
                )
        
        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        idx_tuple = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        return self.data[flat_idx]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        idx_tuple = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        self.data[flat_idx] = float(value)

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        new_shape = validate_shape(new_shape)
        if compute_size(new_shape) != self.size:
            raise ValueError(
                f"Cannot reshape {self.shape} to {new_shape}: sizes differ"
            )
        return DenseTensor(new_shape, data=self.data.copy())

    def unfolding(self, mode: int) -> DenseTensor:
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"Invalid mode: {mode}")
        
        row_count = self.shape[mode]
        col_count = self.size // row_count
        result = DenseTensor.zeros((row_count, col_count))
        
        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            row = multi_idx[mode]
            
            col_multi = multi_idx[:mode] + multi_idx[mode + 1:]
            col_shape = self.shape[:mode] + self.shape[mode + 1:]
            col_strides = compute_strides(col_shape)
            col = multi_index_to_flat(col_multi, col_strides)
            
            result[row, col] = self.data[flat_idx]
        
        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(f"Invalid k: {k}, must be in [0, {self.ndim - 2}]")
        
        left_dims = compute_size(self.shape[:k + 1])
        right_dims = compute_size(self.shape[k + 1:])
        
        result = DenseTensor.zeros((left_dims, right_dims))
        
        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            
            left_idx = 0
            for dim in range(k + 1):
                left_idx = left_idx * self.shape[dim] + multi_idx[dim]
            
            right_idx = 0
            for dim in range(k + 1, self.ndim):
                right_idx = right_idx * self.shape[dim] + multi_idx[dim]
            
            result[left_idx, right_idx] = self.data[flat_idx]
        
        return result

    def copy(self) -> DenseTensor:
        return DenseTensor(self.shape, data=self.data.copy())

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        check_shapes_match(self.shape, other.shape)
        result_data = [a + b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=result_data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        check_shapes_match(self.shape, other.shape)
        result_data = [a - b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=result_data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        scalar = float(scalar)
        result_data = [x * scalar for x in self.data]
        return DenseTensor(self.shape, data=result_data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        return self.__mul__(-1.0)

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        if self.shape != other.shape:
            return False
        
        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        
        return True

    def to_nested_list(self) -> list:
        def build(offset, depth):
            if depth == self.ndim - 1:
                return self.data[offset:offset + self.shape[depth]]
            
            stride = self.strides[depth]
            return [
                build(offset + idx * stride, depth + 1)
                for idx in range(self.shape[depth])
            ]
        
        if self.size == 0:
            return []
        return build(0, 0)

    def __repr__(self) -> str:
        return f"DenseTensor(shape={self.shape}, data={self.to_nested_list()})"

    def __str__(self) -> str:
        return self.__repr__()