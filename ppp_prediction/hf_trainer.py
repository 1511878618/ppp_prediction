from typing import Optional, Tuple
from torch import nn, Tensor
from transformers import Trainer
import torch
from torch.nn import functional as F

from torch.utils.data import RandomSampler, WeightedRandomSampler


class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    comes from : https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = torch.Tensor(alpha) if alpha is not Tensor else alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if y.dim() > 1:  # (N, C) => (N,)
            y = torch.argmax(y, dim=1)
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            if y.dim() > 1:
                y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        # print(log_p.dtype, y.dtype)

        ce = self.nll_loss(log_p, y.long())

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.class_weight = (
            kwargs["class_weight"]
            if len(kwargs["class_weight"]) > 0
            else kwargs["sampling_class_weight"]
        )
        self.sampling_class_weight = kwargs[
            "sampling_class_weight"
        ]  # must have, as auto generated
        self.gamma = kwargs.pop("gamma", 1.0)

        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.3]))
        loss_fct = FocalLoss(
            alpha=torch.tensor(self.class_weight, dtype=torch.float32).to(
                logits.device
            ),
            gamma=torch.tensor(self.gamma, dtype=torch.float32).to(logits.device),
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:

        label = "label"  # only works for cell classification

        # if "class_weight" in self.train_dataset.features.keys():
        #     class_weight = self.train_dataset.features["class_weight"]

        # else:
        #     class_weight = [self.class_weight[i] for i in self.train_dataset[label]]
        sampler = WeightedRandomSampler(
            self.sampling_class_weight, len(self.train_dataset), replacement=True
        )

        return sampler
