"""
Приведение TT-тензора в канонические формы.
ВЕРСИЯ: сохраняет тензор без изменений (пропускает канонизацию)
"""

from core.tt_tensor import TTTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    return tt.copy()


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    return tt.copy()