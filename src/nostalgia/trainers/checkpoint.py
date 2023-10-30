import logging
import re
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Literal, Union

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig

from ..models.models import LLM
from ..trainers.utils import (
    TrainLogging,
    TrainLossTracker,
    broadcast_state_dict,
    is_distributed,
)
from ..utils.git import latest_commit
from ..utils.helper import GenericConfig, create_optimizers, filter_keys_by_part
from ..utils.logging import RankLoggerAdapter


logger = RankLoggerAdapter(logging.getLogger(__name__))
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)


class TrainCheckpoint:
    """
    Helper class for saving model checkpoints, optimizers, and managing best model tracking.

    Args:
        experiment_dir (Union[Path, str]): Directory where experiment data is stored.
        train_config (GenericConfig): Configuration object containing training parameters.

    Attributes:
        logger: Logger instance for printing status messages.
        checkpoint_dir (Path): Directory for storing checkpoints.
        best_model_flag (dict): Dictionary to track best model performance metrics.
        train_config (GenericConfig): Configuration object containing training parameters.

    Methods:
        save_checkpoint(epoch, model, optimizers, schedulers): Saves model, optimizers, and schedulers as checkpoint.
        check_and_save_best_model(epoch, model, optimizers, schedulers, train_stats, validate_stats): Checks if current
            model is better than previous best and saves if necessary.
    """

    def __init__(
        self,
        experiment_dir: Path | str,
        train_config: GenericConfig,
        model: LLM,
        optimizers: dict,
        schedulers: dict,
        training_logger: TrainLogging,
        train_loss_tracker: TrainLossTracker,
        validation_loss_tracker: TrainLossTracker,
        grad_scaler: GradScaler = None,
        rank: int = 0,
        is_peft: bool = False,
    ):
        """
        Initializes the TrainCheckpoint instance.

        Args:
            experiment_dir (Union[Path, str]): Directory where experiment data is stored.
            train_config (GenericConfig): Configuration object containing training parameters.
        """
        self.rank = rank
        self.train_config = train_config
        self.__logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))

        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.grad_scaler = grad_scaler
        self.best_model_flag = {
            "train_loss": torch.tensor(float("inf")),
            "val_loss": torch.tensor(float("inf")),
            "train_metric": torch.tensor(float("inf")),
            "val_metric": torch.tensor(float("inf")),
        }
        self.checkpoint_dir = Path(experiment_dir) / "checkpoints"
        self.best_model_dir = self.checkpoint_dir / "best-model"
        self.is_peft = is_peft
        if self.rank != 0:
            return
        self.__logger.info("Initializing Checkpoint Directories ...")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training_logger = training_logger
        self.train_loss_tracker = train_loss_tracker
        self.validation_loss_tracker = validation_loss_tracker

    def save_checkpoint(self, epoch: int, train_stats: dict, validate_stats: dict) -> None:
        """
        Saves the current model, optimizers, and schedulers as a checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
        """
        is_best_model = False
        best_metric = self.train_config.trainer.best_metric
        if validate_stats["losses"][best_metric] < self.best_model_flag["val_metric"]:
            is_best_model = True
            if self.rank == 0:
                msg = f'Current model with {best_metric} of {validate_stats["losses"][best_metric]:0.4f} has better performance than the current best model with {best_metric} of {self.best_model_flag["val_metric"]:0.4f}'
                self.__logger.info(msg)
            self._update_best_model_flag(train_stats, validate_stats)

        if (epoch + 1) % self.train_config.trainer.save_every != 0:
            if is_best_model:
                self._save_best_model(epoch)
            return
        if is_distributed():
            dist.barrier()
        save_dir = None
        if self.rank == 0:
            self.__logger.info("Creating Checkpointing Directory for Epoch %s", epoch)
            save_dir = self.checkpoint_dir / f"epoch-{epoch}"
            save_dir.mkdir(exist_ok=True)
        self._save_model_state(epoch, save_dir)
        self._save_optimizers_state(epoch, save_dir)
        if is_distributed():
            dist.barrier()
        if self.rank == 0 and is_best_model:
            self.__logger.info("Saving Best Model ...")
            shutil.copytree(save_dir, self.best_model_dir, dirs_exist_ok=True)

    def check_and_save_best_model(self, epoch: int, train_stats: dict, validate_stats: dict) -> None:
        """
        Checks if the current model performance is better than the previous best and saves if necessary.

        Args:
            epoch (int): Current epoch of the training process.
            train_stats (dict): Training statistics.
            validate_stats (dict): Validation statistics.
        """
        best_metric = self.train_config.trainer.best_metric
        if validate_stats[best_metric] < self.best_model_flag["val_metric"]:
            if self.rank == 0:
                msg = f'Current model with {best_metric} of {validate_stats[best_metric]:0.4f} has better performance than the current best model with {best_metric} of {self.best_model_flag["val_metric"]:0.4f}'
                self.__logger.info(msg)

            self._save_best_model(epoch)
            self._update_best_model_flag(train_stats, validate_stats)

    def _save_model_state(self, epoch: int, save_dir: Union[Path, str]):
        """
        Saves the model's state as part of the checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
            save_dir (Union[Path, str]): Directory where the model checkpoint will be saved.
        """
        # TODO: The state of the schedulers is not saved
        file_name = save_dir / "model-checkpoint.pth"
        self.__logger.info("Saving Model State: %s ...", file_name)
        model_type = type(self.model).__name__
        if self.is_peft:
            model_state = self.model.save_pretrained(save_dir)
        else:
            model_state = self.model.state_dict()

        state = {
            "model_type": model_type,
            "last_epoch": epoch,
            "model_state": model_state,
            "params": self.train_config.to_dict(),
            "checkpointer_state": self.state_dict(),
            "training_logger": self.training_logger.state_dict(),
            "loss_trackers": {"training": self.train_loss_tracker.state_dict(), "validation": self.validation_loss_tracker.state_dict()},
            "commit": latest_commit(),
        }
        torch.save(state, file_name)

    def _save_optimizers_state(self, epoch: int, save_dir: Union[Path, str]):
        """
        Saves the state of optimizers and schedulers as part of the checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
            save_dir (Union[Path, str]): Directory where the optimizer checkpoint will be saved.
        """
        file_name = save_dir / "optimizers-checkpoint.pth"
        self.__logger.info("Saving Optimimizers State: %s ...", file_name)
        state = {
            "commit": latest_commit(),
            "last_epoch": epoch,
            "grad_scaler": None if self.grad_scaler is None else self.grad_scaler.state_dict(),
        }
        for name, optimizer in self.optimizers.items():
            schedulers_state = (
                [scheduler.state_dict() for _, scheduler in optimizer["schedulers"]] if optimizer["schedulers"] is not None else None
            )
            state[name] = {"opt": optimizer["opt"].state_dict(), "schedulers": schedulers_state}
        torch.save(state, file_name)

    def _load_optimizers_state(self, load_dir: Union[Path, str]):
        """
        Loads the state of optimizers from a checkpoint.

        Args:
            load_dir (Union[Path, str]): Directory where the optimizer checkpoint is saved.
        """
        file_name = load_dir / "optimizers-checkpoint.pth"
        if not file_name.exists():
            self.__logger.warning("Optimizer checkpoint file %s not found. Skipping optimizer state loading.", file_name)
            return

        self.__logger.info("Loading Optimizers State from %s ...", file_name)
        checkpoint = torch.load(file_name, map_location=torch.device("cpu"))  # Load checkpoint on CPU
        if self.grad_scaler is not None and checkpoint["grad_scaler"] is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        for name, optimizer in self.optimizers.items():
            if name in checkpoint:
                optimizer["opt"].load_state_dict(checkpoint[name]["opt"])
                for ix, (_, scheduler) in enumerate(optimizer["schedulers"]):
                    scheduler.load_state_dict(checkpoint[name]["schedulers"][ix])
                self.__logger.info("Loaded optimizer state for %s.", name)
            else:
                self.__logger.warning("Optimizer state for %s not found in checkpoint.", name)

    def _save_best_model(self, epoch: int):
        """
        Saves the best model based on validation performance.

        Args:
            epoch (int): Current epoch of the training process.
        """
        best_model_dir = None
        if is_distributed():
            dist.barrier()
        if self.rank == 0:
            best_model_dir = self.best_model_dir
            best_model_dir.mkdir(exist_ok=True)
            self.__logger.info("Saving Best Model ...")
        self._save_model_state(epoch, best_model_dir)
        self._save_optimizers_state(epoch, best_model_dir)

    def _update_best_model_flag(self, train_stats: dict, validation_stats: dict) -> None:
        """
        Updates the best model flag with current performance metrics.

        Args:
            train_stats (dict): Training statistics.
            validation_stats (dict): Validation statistics.
        """
        best_metric = self.train_config.trainer.best_metric
        self.best_model_flag["train_loss"] = train_stats["losses"]["loss"]
        self.best_model_flag["val_loss"] = validation_stats["losses"]["loss"]
        self.best_model_flag["train_metric"] = train_stats["losses"][best_metric]
        self.best_model_flag["val_metric"] = validation_stats["losses"][best_metric]

    def load_checkpoint(self, checkpoint: Union[int, Literal["best-model", "last-epoch"]]) -> int:
        """
        Loads a checkpoint for the trainer.

        Args:
            checkpoint (Union[int, Literal["best-model", "last-epoch"]]):
                The checkpoint to load.
                - If an integer is provided, it represents a specific epoch checkpoint.
                - If "best-model" is provided, it loads the best model checkpoint.
                - If "last-epoch" is provided, it loads the checkpoint from the last epoch.

        Returns:
            None
        """
        if not isinstance(checkpoint, (int, str)) or (isinstance(checkpoint, str) and checkpoint not in {"best-model", "last-epoch"}):
            raise ValueError(
                f"Invalid checkpoint value: {checkpoint}. Supported values are 'best-model', 'last-epoch', or an integer epoch number."
            )
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint)
            self.__logger.info("Loading Checkpoint: %s ...", checkpoint_path)
            epoch = self._load_model_state(checkpoint_path)
            self._load_optimizers_state(checkpoint_path)
        except FileNotFoundError as e:
            self.__logger.warning(e)
            if self.is_peft:
                self.__logger.critical("Cannot find chekpoint to resume the training. Start the trainin without '--resume' option!")
                sys.exit(1)
            self.__logger.warning("Starting Training from Scratch")
            return 0

        return epoch

    def _load_model_state(self, checkpoint_path: Path):
        model_checkpoint_path = checkpoint_path / "model-checkpoint.pth"
        self.__logger.info("Loading Model State: %s ...", model_checkpoint_path)
        state = torch.load(model_checkpoint_path)
        if self.is_peft:
            self.__logger.info("Loading PEFT model ...")
            self.model.load_peft_pretrained_model(checkpoint_path)
            self.optimizers = create_optimizers(self.model, self.train_config.optimizers)
        if state["model_state"] is not None:
            self.model.load_state_dict(state["model_state"], strict=not self.is_peft)
        self.training_logger.load_state_dict(state["training_logger"])
        self.train_loss_tracker.load_state_dict(state["loss_trackers"]["training"])
        self.validation_loss_tracker.load_state_dict(state["loss_trackers"]["validation"])
        self.load_state_dict(state["checkpointer_state"])
        return int(state["last_epoch"]) + 1

    def _get_checkpoint_path(self, checkpoint: Union[int, Literal["best-model", "last-epoch"]]) -> Path:
        if checkpoint == "best-model":
            checkpoint_path = self.checkpoint_dir / "best-model"
        elif checkpoint == "last-epoch":
            checkpoint_path = self.__get_last_epoch()
        else:
            checkpoint_path = self.checkpoint_dir / f"epoch-{checkpoint}"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found")
        return checkpoint_path

    def __get_last_epoch(self):
        epoch_numbers = []
        for checkpoint_name in [item.name for item in self.checkpoint_dir.iterdir() if item.is_dir()]:
            match = re.match("^epoch-(\d+)$", checkpoint_name)
            if match:
                epoch_numbers.append(int(match.group(1)))
        if not epoch_numbers:
            raise FileNotFoundError(f"Checkpoints not found in '{self.checkpoint_dir}' not found")
        return self.checkpoint_dir / f"epoch-{max(epoch_numbers)}"

    def state_dict(self) -> dict:
        return self.best_model_flag

    def load_state_dict(self, state: dict):
        self.best_model_flag = state


class TrainCheckpointFSDPFullStateDict(TrainCheckpoint):
    """
    Helper class for saving model checkpoints, optimizers, and managing best model tracking in a FSDP setting.

    Args:
        experiment_dir (Union[Path, str]): Directory where experiment data is stored.
        train_config (GenericConfig): Configuration object containing training parameters.

    Attributes:
        logger: Logger instance for printing status messages.
        checkpoint_dir (Path): Directory for storing checkpoints.
        best_model_flag (dict): Dictionary to track best model performance metrics.
        train_config (GenericConfig): Configuration object containing training parameters.

    Methods:
        save_checkpoint(epoch, model, optimizers, schedulers): Saves model, optimizers, and schedulers as checkpoint.
        check_and_save_best_model(epoch, model, optimizers, schedulers, train_stats, validate_stats): Checks if current
            model is better than previous best and saves if necessary.
    """

    def __init__(self, **kwargs):
        self.__logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        super().__init__(**kwargs)

    def _save_model_state(self, epoch: int, save_dir: Path | str):
        """
        Saves the model's state as part of the checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
            save_dir (Union[Path, str]): Directory where the model checkpoint will be saved.
        """

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
            if self.is_peft:
                self.__logger.info("Getting PEFT Model State dict to rank 0 ...")
                model_state = self.model.get_peft_state_dict()
                if self.train_config.model.name == "HSN":
                    model_state_rest = self.model.state_dict()
                    if self.model.has_common_backbone:
                        model_state_rest = filter_keys_by_part(model_state_rest, "backbone")
                    else:
                        model_state_rest = filter_keys_by_part(model_state_rest, "encoder.backbone")
                        model_state_rest = filter_keys_by_part(model_state_rest, "decoder.backbone")
                    model_state["rest"] = model_state_rest
            else:
                self.__logger.info("Getting Model State dict to rank 0 ...")
                model_state = self.model.state_dict()
            self.__logger.info("Model State dict transferd to rank 0")

        if self.rank == 0:
            file_name = save_dir / "model-checkpoint.pth"
            self.__logger.info("Saving Model State: %s ...", file_name)
            model_type = type(self.model).__name__
            state = {
                "model_type": model_type,
                "last_epoch": epoch,
                "model_state": model_state.get("rest", None) if self.is_peft else model_state,
                "params": self.train_config.to_dict(),
                "checkpointer_state": self.state_dict(),
                "training_logger": self.training_logger.state_dict(),
                "loss_trackers": {
                    "training": self.train_loss_tracker.state_dict(),
                    "validation": self.validation_loss_tracker.state_dict(),
                },
                "commit": latest_commit(),
            }
            torch.save(state, file_name)
            self.__logger.info("Done Saving Model State: %s ...", file_name)
            if self.is_peft:
                if self.train_config.model.name == "LLMCausal":
                    torch.save(model_state["backbone"], save_dir / "adapter_model.bin")
                    self.model.peft_config.save_pretrained(save_dir)
                    self.__logger.info("Done Saving PEFT ...")
                elif self.train_config.model.name == "HSN":
                    for block in ["encoder", "decoder"]:
                        if block in model_state:
                            if self.model.has_common_backbone:
                                self.model.backbone.peft_config[block].save_pretrained(save_dir / block)
                            else:
                                getattr(self.model, block).backbone.peft_config[block].save_pretrained(save_dir / block)
                            torch.save(model_state[block], save_dir / block / "adapter_model.bin")
                    self.__logger.info("Done Saving PEFT ...")
                else:
                    raise ValueError(f"PEFT saving for the model {self.train_config.model.name} is undefined!")

    def _save_optimizers_state(self, epoch: int, save_dir: Path | str):
        """
        Saves the state of optimizers and schedulers as part of the checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
            save_dir (Union[Path, str]): Directory where the optimizer checkpoint will be saved.
        """

        state = {
            "commit": latest_commit(),
            "last_epoch": epoch,
            "grad_scaler": None if self.grad_scaler is None else self.grad_scaler.state_dict(),
        }
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, optim_state_dict_config=optim_save_policy):
            for name, optimizer in self.optimizers.items():
                self.__logger.info("Getting Optimizer State: %s", name)
                optim_state = FSDP.optim_state_dict(self.model, optimizer["opt"])
                if self.rank == 0:
                    schedulers_state = (
                        [scheduler.state_dict() for _, scheduler in optimizer["schedulers"]]
                        if optimizer["schedulers"] is not None
                        else None
                    )
                    state[name] = {"opt": optim_state, "schedulers": schedulers_state}
        if self.rank == 0:
            file_name = save_dir / "optimizers-checkpoint.pth"
            self.__logger.info("Saving Optimimizers State: %s ...", file_name)
            torch.save(state, file_name)
            self.__logger.info("Done Saving Optimimizers State: %s ...", file_name)

    def load_model_state(self, checkpoint: Union[int, Literal["best-model", "last-epoch"]]) -> int:
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint)
        except FileNotFoundError as e:
            self.__logger.warning(e)
            self.__logger.warning("Starting Training from Scratch")
            return 0
        last_epoch = None
        checkpointer_state = None
        if self.rank == 0:
            model_checkpoint_path = checkpoint_path / "model-checkpoint.pth"
            self.__logger.info("Loading Model State: %s ...", model_checkpoint_path)
            state = torch.load(model_checkpoint_path)
            if self.is_peft:
                if state["model_state"] is not None:
                    self.model.load_state_dict(state["model_state"], strict=not self.is_peft)
                self.model.load_peft_pretrained_model(checkpoint_path)
            else:
                self.model.load_state_dict(state["model_state"])

            self.training_logger.load_state_dict(state["training_logger"])
            self.train_loss_tracker.load_state_dict(state["loss_trackers"]["training"])
            self.validation_loss_tracker.load_state_dict(state["loss_trackers"]["validation"])
            last_epoch = int(state["last_epoch"])
            checkpointer_state = state["checkpointer_state"]
        dist.barrier()
        last_epoch = broadcast_state_dict(last_epoch, "last_epoch")
        checkpointer_state = broadcast_state_dict(checkpointer_state, "checkpointer_state", True)
        self.load_state_dict(checkpointer_state)
        return last_epoch + 1

    def load_optimizers_state(self, checkpoint: Union[int, Literal["best-model", "last-epoch"]]):
        """
        Loads the state of optimizers from a checkpoint.

        Args:
            load_dir (Union[Path, str]): Directory where the optimizer checkpoint is saved.
        """
        try:
            checkpoint = self._get_checkpoint_path(checkpoint)
        except FileNotFoundError as e:
            self.__logger.warning(e)
            self.__logger.warning("Starting Training from Scratch")
            return
        file_name = checkpoint / "optimizers-checkpoint.pth"
        if not file_name.exists():
            self.__logger.warning("Optimizer checkpoint file %s not found. Skipping optimizer state loading.", file_name)
            return

        checkpoint_ = {}
        grad_scaler_state = None
        if self.rank == 0:
            self.__logger.info("Loading Optimizers State from %s ...", file_name)
            checkpoint_ = torch.load(file_name)  # Load checkpoint on CPU
            grad_scaler_state = checkpoint_["grad_scaler"]
        grad_scaler_state = broadcast_state_dict(grad_scaler_state, "grad_scaler")
        dist.barrier()
        if self.grad_scaler is not None and grad_scaler_state is not None:
            self.grad_scaler.load_state_dict(grad_scaler_state)
        for name, optimizer in self.optimizers.items():
            schedulers_sd = [None] * len(optimizer["schedulers"])
            full_osd = None
            if name in checkpoint_:
                full_osd = checkpoint_[name]["opt"]
                schedulers_sd = checkpoint_[name]["schedulers"]
            sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, self.model)
            optimizer["opt"].load_state_dict(sharded_osd)
            self.__logger.info("Optimizer %s Loaded!", name)
            for ix, scheduler_sd in enumerate(schedulers_sd):
                scheduler_sd = broadcast_state_dict(scheduler_sd, "scheduler_state_dict")
                # self.__logger.debug("Recieved Scheduler state for optimizer '%s': %s!", name, schedulers_sd)
                optimizer["schedulers"][ix][1].load_state_dict(scheduler_sd)
            self.__logger.info("Optimizer-Schedulers %s Loaded!", name)


non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model, check_fn: callable):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    logger.info("Applying FSDP Activation Checkpointing ...")

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
