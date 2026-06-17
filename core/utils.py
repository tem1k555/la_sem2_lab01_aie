"""Вспомогательные функции для работы с тензорами."""

from typing import Tuple, List, Union


def validate_shape(
    shape: tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    """
    Проверяет корректность формы тензора и приводит её к стандартному виду.

    Убеждается, что shape является последовательностью положительных целых
    чисел. Преобразует список в кортеж для единообразия.

    Args:
        shape: кортеж или список размеров тензора по каждой моде

    Returns:
        tuple: проверенный кортеж положительных целых чисел

    Raises:
        TypeError:  если shape не является tuple или list
        ValueError: если хотя бы один элемент shape не является
                    положительным целым числом
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"Shape must be tuple or list, got {type(shape)}")
    
    if len(shape) == 0:
        raise ValueError("Shape cannot be empty")
    
    result = tuple(shape)
    
    for i, dim in enumerate(result):
        if not isinstance(dim, int):
            raise ValueError(f"Dimension {i} must be int, got {type(dim)}")
        if dim <= 0:
            raise ValueError(f"Dimension {i} must be positive, got {dim}")
    
    return result


def compute_size(shape: tuple[int, ...]) -> int:
    """
    Возвращает общее число элементов тензора заданной формы.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    result = 1
    for dim in shape:
        result *= dim
    return result


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает кортеж strides, содержащий для каждой моды k свои strides[k].

    Stride по моде k — это число элементов в плоском списке, на которое
    нужно сдвинуться, чтобы перейти к следующему элементу вдоль моды k.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    
    Используется row-major (C-order) порядок:
    - последняя мода меняется быстрее всего
    - stride для последней моды = 1
    - stride[k] = shape[k+1] * shape[k+2] * ... * shape[d-1]
    """
    d = len(shape)
    strides = [1] * d
    
    # Вычисляем strides в row-major порядке
    # Для последней моды stride = 1
    for k in range(d - 2, -1, -1):
        strides[k] = strides[k + 1] * shape[k + 1]
    
    return tuple(strides)


def multi_index_to_flat(
    multi_index: tuple[int, ...],
    strides: tuple[int, ...]
) -> int:
    """
    Возвращает позицию элемента в плоском списке данных по его
    многомерным координатам и заранее вычисленным strides.

    Args:
        multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1})
        strides:     кортеж шагов   (s_0, s_1, ..., s_{d-1})
    
    Формула: flat_index = sum(multi_index[k] * strides[k])
    """
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
    """
    Возвращает мультииндекс на основе плоского индекса.

    Args:
        flat_index: плоский индекс в списке данных
        shape:      кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    
    Используется row-major (C-order) порядок.
    """
    d = len(shape)
    multi_index = [0] * d
    
    # Вычисляем strides в row-major порядке
    strides = compute_strides(shape)
    
    remaining = flat_index
    for k in range(d):
        multi_index[k] = remaining // strides[k]
        remaining = remaining % strides[k]
    
    return tuple(multi_index)


def check_shapes_match(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...]
) -> None:
    """
    Проверяет совпадение форм двух тензоров.

    Используется перед поэлементными операциями (сложение, вычитание),
    чтобы гарантировать совместимость тензоров.

    Args:
        shape1: кортеж размеров первого тензора
        shape2: кортеж размеров второго тензора

    Raises:
        ValueError: если формы не совпадают
    """
    if shape1 != shape2:
        raise ValueError(
            f"Shapes must match: {shape1} vs {shape2}"
        )