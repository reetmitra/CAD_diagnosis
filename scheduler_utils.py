import math
import copy
import torch


class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    """Thin wrapper around PyTorch's CosineAnnealingWarmRestarts.

    Useful for escaping local minima during long fine-tuning runs.  Each
    restart resets the LR to ``base_lr`` (but optionally scaled by
    ``t_mult`` so each cycle grows longer).

    Args:
        optimizer: Wrapped optimizer.
        t_0: Number of epochs for the first cycle.
        t_mult: Multiplicative factor for cycle length after each restart (default: 2).
        eta_min: Minimum LR at trough of cosine (default: 1e-7).
        warmup_epochs: Linear warmup before the first cosine cycle (default: 0).
    """

    def __init__(self, optimizer, t_0, t_mult=2, eta_min=1e-7, warmup_epochs=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self._base_lrs_stored = [pg['lr'] for pg in optimizer.param_groups]
        super().__init__(optimizer, T_0=max(1, t_0 - warmup_epochs), T_mult=t_mult,
                         eta_min=eta_min, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self._base_lrs_stored]
        return super().get_lr()


class LinearWarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    """LR scheduler with linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        warmup_epochs: int = 10,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]


class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        source = model.module if hasattr(model, "module") else model
        for name, param in source.state_dict().items():
            self.shadow[name] = param.clone().detach()

    def update(self, model: torch.nn.Module) -> None:
        source = model.module if hasattr(model, "module") else model
        for name, param in source.state_dict().items():
            if name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()
                )

    def apply(self, model: torch.nn.Module) -> None:
        source = model.module if hasattr(model, "module") else model
        self.backup = {}
        for name, param in source.state_dict().items():
            if name in self.shadow:
                self.backup[name] = param.clone().detach()
        source.load_state_dict(self.shadow, strict=False)

    def restore(self, model: torch.nn.Module) -> None:
        source = model.module if hasattr(model, "module") else model
        source.load_state_dict(self.backup, strict=False)
        self.backup = {}


def build_param_groups(
    model: torch.nn.Module, base_lr: float
) -> list[dict[str, object]]:
    """Build optimizer param groups with per-component learning rates.

    Three tiers:
      - heads (1.0× LR)   : classification / box-regression heads
      - transformer (0.5×) : all attention / encoder / decoder layers and
                             positional / query embeddings
      - backbone (0.1×)   : all 3-D and 2-D CNN feature-extraction blocks

    Parameter assignment uses substring matching on the full dotted name as
    returned by ``model.named_parameters()``.  Head and transformer checks are
    evaluated first; anything unmatched falls to backbone.
    """

    # SC-Net specific patterns (matched against full parameter name)
    head_keywords = (
        "softmax_classify",     # temporal SC classification head (MLP)
        "object_detection",     # spatial OD heads (class + box MLPs)
    )
    transformer_keywords = (
        "temporal_correlation_analysis",  # TransformerEncoderLayer stack
        "transformer_architecture",       # nn.Transformer in spatial branch
        "query_pos",                      # DETR-style learnable query embeddings
        "pos_embedding",                  # learned positional encoding (temporal)
    )
    # Everything else is a CNN backbone block

    backbone_params: list[torch.nn.Parameter] = []
    transformer_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in head_keywords):
            head_params.append(param)
        elif any(k in name for k in transformer_keywords):
            transformer_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": 0.1 * base_lr, "name": "backbone"})
    if transformer_params:
        param_groups.append({"params": transformer_params, "lr": 0.5 * base_lr, "name": "transformer"})
    if head_params:
        param_groups.append({"params": head_params, "lr": 1.0 * base_lr, "name": "heads"})

    return param_groups
