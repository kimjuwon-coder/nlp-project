import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BaseModel


class EarlyStopping:
    def __init__(self, val_key, tolerance=5, higher_better=True):
        self.deviation = 0
        self.val_key = val_key
        self.tolerance = tolerance
        self.higher_better = higher_better

        if higher_better is True:
            self.best_standard = float("-inf")

        else:
            self.best_standard = float("inf")

        return

    def check(self, postfix: dict[str, float]) -> tuple[bool, bool]:
        early_stop = False
        standard = postfix[self.val_key]
        improved = self.improvement_check(standard)

        if improved is True:
            self.deviation = 0
            self.best_standard = standard

        else:
            self.deviation += 1
            if self.deviation > self.tolerance:
                early_stop = True

        return early_stop, improved

    def improvement_check(self, standard: float) -> bool:
        if self.higher_better is True:
            return standard > self.best_standard

        else:
            return standard < self.best_standard


@torch.no_grad()
def val_loop(model: BaseModel, val_loader: DataLoader) -> dict[str, float]:
    device = next(model.parameters()).device
    val_metrics = {key: [] for key in model.val_keys}
    dataset_len = len(val_loader.dataset)

    model.eval()
    for batch in val_loader:
        batch: dict[str, Tensor]
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

        batch_metrics = model.validate_batch(**batch)
        for key, val in batch_metrics.items():
            val_metrics[key].append(val)
            # val: reduction `sum` applied

    val_metrics = {
        f"val {key}": sum(results) / dataset_len for key, results in val_metrics.items()
    }

    return val_metrics


@torch.no_grad()
def test(model: BaseModel, test_loader: DataLoader) -> dict[str, float]:
    device = next(model.parameters()).device
    test_metrics = {key: [] for key in model.val_keys}
    dataset_len = len(test_loader.dataset)
    outputs = []

    model.eval()
    for batch in test_loader:
        batch: dict[str, Tensor]
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

        output = model.predict(**batch)
        outputs.append(output)

        test_metrics = model.validate_batch(**batch)
        for key, val in test_metrics.items():
            test_metrics[key] = val

    test_metrics = {
        f"test {key}": sum(results) / dataset_len
        for key, results in test_metrics.items()
    }

    # example for Time Series
    # outputs = torch.cat(outputs, dim=0)
    # outputs = torch.cat([outputs[:, 0], outputs[-1, 1:]], dim=0)

    return outputs, test_metrics


def pbar_finish(
    pbar: tqdm,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    formatter_train: dict[str, str] = None,
    formatter_val: dict[str, str] = None,
) -> dict[str, float]:
    # train_metrics = {
    #     key: formatter_train[f"train {key}"].format(val) \
    #         for key, val in train_metrics.items()
    # }
    # val_metrics = {
    #     key: formatter_val[f"val {key}"].format(val) \
    #         for key, val in val_metrics.items()
    # }
    postfix = train_metrics | val_metrics
    pbar.set_postfix(postfix)
    pbar.close()

    return postfix
