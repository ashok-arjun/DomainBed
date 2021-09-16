import torch
import torch.nn as nn


from typing import Dict, Tuple, Any, Union, Optional, Callable, Iterable, List
from dataclasses import dataclass

from mask_utils import ParameterPointer, append_update, Masks
from gumbel_sigmoid import gumbel_sigmoid

from grad_norm import GradNormTracker

from models import ConvNet

class MaskedModel(torch.nn.Module):
    def gather_and_remove_params(self, module: torch.nn.Module) -> Tuple[Dict[str, ParameterPointer],
                                                                         Dict[str, torch.nn.Parameter]]:
        res_ptrs, res_params = {}, {}

        for name, m in module.named_children():
            ptrs, params = self.gather_and_remove_params(m)
            append_update(res_ptrs, ptrs, name)
            append_update(res_params, params, name)

        none_params = []
        for name, param in module._parameters.items():
            if param is None:
                none_params.append(name)
                continue

            res_ptrs[name] = ParameterPointer(module, name, False)
            res_params[name] = param

        # module._parameters.clear()
        # for n in none_params:
        #     module._parameters[n] = None

        return res_ptrs, res_params

    def sample_mask(self, mask: torch.Tensor, n_samples: int) -> torch.Tensor:
        if n_samples > 0:
            if n_samples > 1:
                mask = mask.unsqueeze(0).expand(n_samples, *mask.shape)
            return gumbel_sigmoid(mask, hard=True)
        else:
            return (mask >= 0).float()

    def _count_params(self, params: Iterable[torch.Tensor]) -> int:
        return sum(p.numel() for p in params)

    def __init__(self, model: torch.nn.Module, n_mask_sets: int = 1, n_mask_samples: int = 1, mask_loss_weight: float = 1e-4,
                 mask_filter: Callable[[str], bool] = lambda x: True, empty_init: float = 1):
        super().__init__()

        self.pointers, params = self.gather_and_remove_params(model)
        self.param_names = list(sorted(self.pointers.keys()))
        self.pointer_values = [self.pointers[n] for n in self.param_names]
        self.model_parameters = torch.nn.ParameterDict(params)
        self.masks = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.full_like(v, empty_init))
                                         for k, v in self.model_parameters.items() if mask_filter(k)})

        self.masked_params = set(self.masks.keys())

        self.n_mask_samples = n_mask_samples
        self.mask_loss_weight = mask_loss_weight
        self.active = 1
        self.temporary_masks: Optional[Masks] = None

        print(f"Found module parameters: {list(self.model_parameters.keys())}")
        print(f"Masking is applied to paramteres: {self.masked_params}")

        n_total = self._count_params(self.model_parameters.values())
        n_masked = self._count_params(self.masks.values())

        print(f"Masking {n_masked} out of {n_total} parameters ({n_masked*100/n_total:.1f} %)")

        single_sample_params = [k for k in self.masked_params if not self.pointers[k].multimask_support]

        if single_sample_params:
            is_ok = n_mask_samples == 1
            print(f"!!!!!!!!!!!!!!!!!!!!!!!! {'WARNING' if is_ok else 'ERROR'} !!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"The following parameters support only single masks {single_sample_params}.")
            assert is_ok

        self.model = model

    @property
    def masking_enabled(self):
        return self.active >= 0

    def get_mask(self, name: str) -> torch.Tensor:
        if self.temporary_masks is not None:
            return self.temporary_masks[name].float()
        else:
            return self.sample_mask(self.masks[name], self.n_mask_samples if self.training else 0)

    def is_masked(self, name: str):
        return name in self.masked_params and  (self.temporary_masks is None or name in self.temporary_masks)

    def update_params(self):
        for name, ptr in self.pointers.items():
            if self.masking_enabled and self.is_masked(name):
                ptr.set(self.model_parameters[name] * self.get_mask(name))
            else:
                ptr.set(self.model_parameters[name])

    def __call__(self, *args, **kwargs):
        if self.masking_enabled:
            self.update_params()

        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.update_params()
        return self

    def mask_loss(self, scales: Optional[Dict[str, float]] = None) -> Union[torch.Tensor, float]:
        if not self.masking_enabled:
            return 0.0

        res = 0.0
        for n, p in self.masks.named_parameters():
            res = res + p.sum()

        return self.mask_loss_weight * res

    def train(self, mode: bool = True):
        res = super().train(mode)
        self.model.train(mode)
        return res

    def set_model_to_eval(self, eval: bool = True):
        self.model.eval()

if __name__ == "__main__":

    from utils import set_seed

    set_seed(0)

    x = torch.randn(4, 10).cuda()
    y = torch.randint(0, 5, size=(4, )).cuda()

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.Linear(20, 5)
    )
    masked_model = MaskedModel(model)
    masked_model.cuda()

    print("PREVIOUS MASKS")
    prev_masks = list(masked_model.masks.parameters())
    print(prev_masks)

    output = masked_model(x)

    optimizer_model = torch.optim.SGD(masked_model.model.parameters(), lr=0.01)
    optimizer_mask = torch.optim.SGD(masked_model.masks.parameters(), lr=1)

    model_loss = nn.CrossEntropyLoss()
    loss = model_loss(output, y)
    print(loss)

    mask_loss = masked_model.mask_loss()
    loss += mask_loss
    print(mask_loss)

    loss.backward()

    optimizer_model.step()
    optimizer_mask.step()

    print("NEW MASKS")
    new_masks = list(masked_model.masks.parameters())
    print(new_masks)


    # Alternative implementation: why not keep masks separate, model separate. Their LR & optim separate.
    # After model forward pass, sample mask using n_samples, and then multiply.
    # Loss backward one-by-one, update both.