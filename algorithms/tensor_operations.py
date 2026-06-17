"""
Базовые операции с TT-тензорами.
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
    """Поэлементное сложение двух TT-тензоров."""
    if tt1.order == 1:
        data = [x + y for x, y in zip(tt1.cores[0].data, tt2.cores[0].data)]
        return TTTensor([DenseTensor(tt1.cores[0].shape, data)])

    cores = []
    for k in range(tt1.order):
        a = tt1.cores[k]
        b = tt2.cores[k]
        a_l, n, a_r = a.shape
        b_l, _, b_r = b.shape

        if k == 0:
            core = DenseTensor.zeros((1, n, a_r + b_r))
            for i in range(n):
                for ar in range(a_r):
                    core[0, i, ar] = a[0, i, ar]
                for br in range(b_r):
                    core[0, i, a_r + br] = b[0, i, br]
        elif k == tt1.order - 1:
            core = DenseTensor.zeros((a_l + b_l, n, 1))
            for i in range(n):
                for al in range(a_l):
                    core[al, i, 0] = a[al, i, 0]
                for bl in range(b_l):
                    core[a_l + bl, i, 0] = b[bl, i, 0]
        else:
            core = DenseTensor.zeros((a_l + b_l, n, a_r + b_r))
            for al in range(a_l):
                for i in range(n):
                    for ar in range(a_r):
                        core[al, i, ar] = a[al, i, ar]
            for bl in range(b_l):
                for i in range(n):
                    for br in range(b_r):
                        core[a_l + bl, i, a_r + br] = b[bl, i, br]

        cores.append(core)

    return TTTensor(cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """Умножение TT-тензора на скаляр."""
    cores = [core.copy() for core in tt.cores]
    cores[0] = backend.scale(cores[0], alpha)
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """Поэлементное произведение (Адамара) двух TT-тензоров."""
    cores = []
    for k in range(tt1.order):
        a = tt1.cores[k]
        b = tt2.cores[k]
        a_l, n, a_r = a.shape
        b_l, _, b_r = b.shape
        core = DenseTensor.zeros((a_l * b_l, n, a_r * b_r))

        for al in range(a_l):
            for bl in range(b_l):
                left = al * b_l + bl
                for i in range(n):
                    for ar in range(a_r):
                        for br in range(b_r):
                            right = ar * b_r + br
                            core[left, i, right] = a[al, i, ar] * b[bl, i, br]

        cores.append(core)

    return TTTensor(cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """Скалярное произведение двух TT-тензоров."""
    env = [[1.0]]

    for k in range(tt1.order):
        a = tt1.cores[k]
        b = tt2.cores[k]
        a_l, n, a_r = a.shape
        b_l, _, b_r = b.shape
        new_env = [[0.0] * b_r for _ in range(a_r)]

        for al in range(a_l):
            for bl in range(b_l):
                coeff = env[al][bl]
                for i in range(n):
                    for ar in range(a_r):
                        aval = a[al, i, ar]
                        for br in range(b_r):
                            new_env[ar][br] += coeff * aval * b[bl, i, br]

        env = new_env

    return env[0][0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """Фробениусова норма TT-тензора."""
    return math.sqrt(max(tt_dot(tt, tt, backend), 0.0))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """Норма разности двух TT-тензоров."""
    value = tt_dot(tt1, tt1, backend) + tt_dot(tt2, tt2, backend) - 2 * tt_dot(tt1, tt2, backend)
    return math.sqrt(max(value, 0.0))