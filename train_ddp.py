import os
from statistics import mean

import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, distributed
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from models import BaseModel
from tools import EarlyStopping, pbar_finish, val_loop

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def train_loop(
    rank,
    epoch,
    check_unit,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
) -> tuple[tqdm, dict] | tuple[None, None]:
    device = model.device

    if rank == 0:
        train_loss = []
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch:2d}")
        pbar_update = pbar.update

    model.train()
    for batch in train_loader:
        batch: dict[str, Tensor]
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

        optimizer.zero_grad()
        loss: Tensor = model(**batch)
        loss.backward()
        optimizer.step()

        if rank == 0:
            train_loss.append(loss.item())
            if pbar.n % check_unit == 0:
                pbar.set_postfix({"loss": mean(train_loss)})
            pbar_update()

    if rank == 0:
        train_metrics = {"train loss": mean(train_loss)}
        return pbar, train_metrics

    else:
        return None, None


def _ddp_setup(rank, world_size, config: DictConfig, train_set, val_set):
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:12355",
        rank=rank,
        world_size=world_size,
    )

    train_sampler = DistributedSampler(train_set, world_size, rank)
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,  # pin_memory=True,
        **config.dataloader.train,
    )

    device = config.gpu_list[rank]
    check_unit = int(len(train_loader) * config.check_prop)
    early_stop = torch.tensor(False).to(device)

    if rank == 0:
        val_loader = DataLoader(val_set, **config.dataloader.val)
        early_stopping = EarlyStopping(**config.early_stopping)

        if config.wandb.do is True:
            run = wandb.init(
                config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
                **config.wandb.kwargs,
            )

    else:
        val_loader = None
        early_stopping = None
        run = None

    return (
        train_loader,
        val_loader,
        early_stopping,
        early_stop,
        check_unit,
        device,
        run,
    )


def _ddp_cleanup():
    distributed.destroy_process_group()
    return


def _ddp_broadcast(tensor: Tensor):
    distributed.barrier()
    distributed.broadcast(tensor, src=0)
    return


def _ddp_train_worker(
    rank: int,
    world_size: int,
    model: type[BaseModel],
    config: DictConfig,
    train_set: Dataset,
    val_set: Dataset,
):
    train_loader, val_loader, early_stopping, early_stop, check_unit, device, run = _ddp_setup(
        rank, world_size, config, train_set, val_set
    )
    model = model(**config.model).to(device)
    model: DDP = DDP(model, device_ids=[device])
    optimizer = torch.optim.Adam(model.parameters(), **config.optimizer)

    for epoch in range(config.epochs):
        pbar, train_metrics = train_loop(rank, epoch, check_unit, model, optimizer, train_loader)
        if rank == 0:
            val_metrics = val_loop(model.module, val_loader)
            postfix = pbar_finish(pbar, train_metrics, val_metrics)

            if config.wandb.do is True:
                run.log(postfix, step=epoch)

            early_stop, improved = early_stopping.check(postfix)
            early_stop = torch.tensor(early_stop, dtype=torch.bool, device=device)

            if improved is True:
                torch.save(
                    model.state_dict(),
                    f"{config.model_save_path}/{config.model_name}_epoch{epoch:2d}.pth",
                )

        _ddp_broadcast(early_stop)  # early_stop: rank 0 -> all ranks
        if early_stop.item() is True:
            break

    _ddp_cleanup()
    if (rank == 0 and config.wandb.do) is True:
        run.finish()

    return


def train(
    model: type[BaseModel],
    config: DictConfig,
    train_set: Dataset,
    val_set: Dataset,
):
    args = (config.ddp.kwargs.nprocs, model, config, train_set, val_set)
    mp.spawn(_ddp_train_worker, args=args, **config.ddp.kwargs)
    return
