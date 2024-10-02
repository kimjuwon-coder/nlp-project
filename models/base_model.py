from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    val_keys: tuple[str] = ()

    def __init__(self):
        super().__init__()
        return

    @abstractmethod
    def get_output(self, *args, **kwargs) -> Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def predict(self, *args, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> dict[str, Tensor | float]:
        pass

    @abstractmethod
    @torch.no_grad()
    def validate_batch(self, *args, **kwargs) -> dict[str, float]:
        pass
