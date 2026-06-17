"""
Вспомогательные функции для работы с тензорами.
"""

from typing import Tuple, List, Union


def validate_shape(
    shape: tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    """
    Проверяет корректность формы тензора и приводит её к стандартному виду.
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"Shape must be tuple or list, got {type(shape)}")
    
    if len(shape) == 0:
        raise ValueError("Shape cannot be empty")
    
    result = tuple(shape)
    
    for dim_idx, dim_size in enumerate(result):
        if not isinstance(dim_size, int):
            raise ValueError(f"Dimension {dim_idx} must be int, got {type(dim_size)}")
        if dim_size <= 0:
            raise ValueError(f"Dimension {dim_idx} must be positive, got {dim_size}")
    
    return result


def compute_size(shape: tuple[int, ...]) -> int:
    """Возвращает общее число элементов тензора заданной формы."""
    result = 1
    for dim_size in shape:
        result *= dim_size
    return result


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает кортеж strides для row-major порядка.
    """
    tensor_order = len(shape)
    strides = [1] * tensor_order
    
    for mode_idx in range(tensor_order - 2, -1, -1):
        strides[mode_idx] = strides[mode_idx + 1] * shape[mode_idx + 1]
    
    return tuple(strides)


def multi_index_to_flat(
    multi_index: tuple[int, ...],
    strides: tuple[int, ...]
) -> int:
    """Возвращает плоский индекс по мультииндексу и strides."""
    if len(multi_index) != len(strides):
        raise ValueError(
            f"Length of multi_index ({len(multi_index)}) must match "
            f"length of strides ({len(strides)})"
        )
    
    flat_index = 0
    for idx, stride in zip(multi_index, strides):
        flat_index += idx * stride
    
    return flat_index


def flat_to_multi_index(
    flat_index: int,
    shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Возвращает мультииндекс по плоскому индексу."""
    tensor_order = len(shape)
    multi_index = [0] * tensor_order
    
    remaining = flat_index
    
    for mode_idx in range(tensor_order):
        stride = compute_strides(shape)[mode_idx]
        multi_index[mode_idx] = remaining // stride
        remaining = remaining % stride
    
    return tuple(multi_index)


def check_shapes_match(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...]
) -> None:
    """Проверяет совпадение форм двух тензоров."""
    if shape1 != shape2:
        raise ValueError(
            f"Shapes must match: {shape1} vs {shape2}"
        )