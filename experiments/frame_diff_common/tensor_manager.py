import hashlib
import torch
from typing import Tuple, Optional


class TensorManager:
    def __init__(self, seed: int = 42, device: str = "cpu"):
        self.seed = seed
        try:
            self.device = torch.device(device)
            # Verify device is available by creating a small tensor
            torch.zeros(1, device=self.device)
        except RuntimeError:
            self.device = torch.device("cpu")

    def generate(self, name: str, iteration: int, shape: Tuple[int, ...],
                 dtype: torch.dtype = torch.float32, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        seed_str = f"{self.seed}_{name}_{iteration}"
        det_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        orig_state = torch.get_rng_state()
        torch.manual_seed(det_seed)
        try:
            if dtype == torch.int64 or dtype == torch.int32:
                tensor = torch.randint(int(low), int(high), shape, dtype=dtype)
            elif dtype == torch.bool:
                tensor = torch.randint(0, 2, shape, dtype=torch.int64).bool()
            else:
                tensor = torch.rand(shape, dtype=dtype) * (high - low) + low
            return tensor.to(self.device)
        finally:
            torch.set_rng_state(orig_state)

    def generate_input_ids(self, name: str, iteration: int, shape: Tuple[int, ...]) -> torch.Tensor:
        return self.generate(name, iteration, shape, dtype=torch.int64, low=0, high=50000)
