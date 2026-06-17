"""
Тензор в TT-формате (Tensor Train).
"""

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size
from core.linalg import matmul


class TTTensor:
    """
    Тензор в TT-формате.
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    def __init__(self, cores: list[DenseTensor]) -> None:
        if not cores:
            raise ValueError("TT-tensor must have at least one core")
        
        self.cores = [core.copy() for core in cores]
        self.order = len(cores)
        
        ranks = [1]
        shape = []
        
        for core_idx, core in enumerate(cores):
            if core.ndim != 3:
                raise ValueError(f"Core {core_idx} must be 3D, got {core.ndim}D")
            
            left_rank, mode_size, right_rank = core.shape
            
            if core_idx == 0 and left_rank != 1:
                raise ValueError(f"First core must have r_0 = 1, got {left_rank}")
            
            if core_idx == len(cores) - 1 and right_rank != 1:
                raise ValueError(f"Last core must have r_d = 1, got {right_rank}")
            
            if core_idx > 0 and left_rank != ranks[-1]:
                raise ValueError(
                    f"Core {core_idx} has left_rank = {left_rank}, "
                    f"but previous core has right_rank = {ranks[-1]}"
                )
            
            ranks.append(right_rank)
            shape.append(mode_size)
        
        self.shape = tuple(shape)
        self.ranks = tuple(ranks)
        
        if self.ranks[0] != 1 or self.ranks[-1] != 1:
            raise ValueError(f"Ranks must start and end with 1, got {self.ranks}")

    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.
        """
        import random
        
        if seed is not None:
            random.seed(seed)
        
        shape = validate_shape(shape)
        tensor_order = len(shape)
        
        if len(ranks) == tensor_order - 1:
            ranks = [1] + list(ranks) + [1]
        elif len(ranks) == tensor_order + 1:
            ranks = list(ranks)
        else:
            raise ValueError(
                f"ranks must have length {tensor_order - 1} (internal ranks) "
                f"or {tensor_order + 1} (full ranks), got {len(ranks)}"
            )
        
        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError(f"First and last ranks must be 1, got {ranks[0]} and {ranks[-1]}")
        
        cores = []
        for mode_idx in range(tensor_order):
            left_rank = ranks[mode_idx]
            right_rank = ranks[mode_idx + 1]
            mode_size = shape[mode_idx]
            
            core = DenseTensor.random((left_rank, mode_size, right_rank), seed=seed)
            cores.append(core)
        
        return TTTensor(cores)

    def get_element(self, indices):
        if len(indices) != self.order:
            raise ValueError(f"Expected {self.order} indices, got {len(indices)}")
        
        for mode_idx, idx in enumerate(indices):
            if idx < 0 or idx >= self.shape[mode_idx]:
                raise IndexError(f"Index {idx} out of bounds for dimension {mode_idx}")
        
        left_vector = DenseTensor((1, self.ranks[0]), data=[1.0])
        
        for mode_idx, idx in enumerate(indices):
            core = self.cores[mode_idx]
            left_rank, mode_size, right_rank = core.shape
            
            slice_matrix = DenseTensor((left_rank, right_rank))
            for row in range(left_rank):
                for col in range(right_rank):
                    slice_matrix[row, col] = core[row, idx, col]
            
            left_vector = matmul(left_vector, slice_matrix)
        
        return left_vector[0, 0]

    def full(self):
        first_core = self.cores[0]
        result = first_core.reshape((first_core.shape[1], first_core.shape[2]))
        
        current_shape = [self.shape[0]]
        
        for mode_idx in range(1, self.order):
            core = self.cores[mode_idx]
            left_rank, mode_size, right_rank = core.shape
            
            result = result.reshape((compute_size(tuple(current_shape)), left_rank))
            unfolded_core = core.reshape((left_rank, mode_size * right_rank))
            result = matmul(result, unfolded_core)
            current_shape.append(mode_size)
            result = result.reshape(tuple(current_shape) + (right_rank,))
        
        return result.reshape(self.shape)

    def core_sizes(self):
        return [core.shape for core in self.cores]

    def total_storage(self):
        return sum(core.size for core in self.cores)

    def compression_ratio(self):
        full_size = compute_size(self.shape)
        tt_size = self.total_storage()
        if tt_size == 0:
            return float('inf')
        return full_size / tt_size

    def copy(self):
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self):
        lines = [
            f"TTTensor(order={self.order}, shape={self.shape})",
            f"  ranks: {self.ranks}",
            f"  core shapes: {self.core_sizes()}",
            f"  total storage: {self.total_storage()} elements",
            f"  compression ratio: {self.compression_ratio():.2f}x",
        ]
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()