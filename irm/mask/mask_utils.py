import torch
from typing import Dict, Tuple, Any, Union, Optional, Callable, Iterable, List
from dataclasses import dataclass

@dataclass
class ParameterPointer:
    parent: torch.nn.Module
    name: str
    multimask_support: bool

    def set(self, data: torch.Tensor):
        self.parent.__dict__[self.name] = data

    def get(self) -> torch.Tensor:
        return self.parent.__dict__[self.name]


def append_update(target: Dict[str, Any], src: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    target.update({f"{prefix}_{k}": v for k, v in src.items()})


class Masks(dict):
    def invert(self, filter = lambda k: True):
        return Masks({k: ~v if filter(k) else v for k, v in self.items()})

    def __or__(self, other):
        res = Masks()
        res.update({k: torch.logical_or(other[k], v) if k in other else v for k, v in self.items()})
        res.update({k: v for k, v in other.items() if k not in self})
        return res

    def __and__(self, other):
        return Masks({k: torch.logical_and(other[k], v) for k, v in self.items() if k in other})