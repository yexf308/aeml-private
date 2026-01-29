from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DatasetBatch:
    samples: torch.Tensor
    local_samples: torch.Tensor
    mu: torch.Tensor
    cov: torch.Tensor
    p: torch.Tensor
    weights: torch.Tensor
    hessians: torch.Tensor

    def as_tuple(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.samples,
            self.local_samples,
            self.mu,
            self.cov,
            self.p,
            self.weights,
            self.hessians,
        )

    @classmethod
    def from_tuple(cls, tensors: Tuple[torch.Tensor, ...]) -> "DatasetBatch":
        if len(tensors) != 7:
            raise ValueError("Expected a tuple of length 7.")
        return cls(*tensors)
