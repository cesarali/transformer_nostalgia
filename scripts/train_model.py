#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import logging
import os
from pathlib import Path
from typing import List

import click
import numpy as np
import torch
import torch.distributed as dist
from nostalgia.data.dataloaders import DataLoaderFactory
from nostalgia.data.dataloaders.reasoning import load_tokenizer
from nostalgia.models import ModelFactory
from nostalgia.trainers.trainer import TrainerFactory
from nostalgia.trainers.utils import (
    cleanup,
    clear_gpu_cache,
    setup,
    setup_environ_flags,
)
from nostalgia.utils.helper import GenericConfig, expand_params, load_yaml
from nostalgia.utils.logging import RankLoggerAdapter, setup_logging


setup_logging()
import warnings


warnings.filterwarnings("ignore", module="matplotlib")
logger = RankLoggerAdapter(logging.getLogger(__name__))


@click.command()
@click.option("-c", "--config", "cfg_path", required=True, type=click.Path(exists=True), help="path to config file")
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-r",
    "--resume",
    "resume",
    is_flag=True,
    default=False,
    show_default=True,
    help="Resume the training from the last checkpoint of the experiment. In case there is no checkpoint it starts new training.",
)
def main(cfg_path: Path, log_level: int, resume: bool):
    config = load_yaml(cfg_path)
    gs_configs = expand_params(config)
    train(gs_configs, resume)


def train(configs: List[GenericConfig], resume: bool):
    for config in configs:
        if config.distributed.enabled:
            train_distributed(config, resume)
        else:
            train_single(config, resume)


def train_distributed(config: List[GenericConfig], resume: bool):
    torch.manual_seed(int(config.experiment.seed))
    torch.cuda.manual_seed(int(config.experiment.seed))
    np.random.seed(int(config.experiment.seed))
    setup()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        logger.info("Starting Experiment: %s", config.experiment.name)
        logger.info("World Size: %d", world_size)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
        device_map = config.experiment.device_map

        tokenizer = load_tokenizer(**config.tokenizer.__dict__)
        dataloader = DataLoaderFactory.create(**config.dataset.__dict__, tokenizer=tokenizer)
        num_added_tokens = len(tokenizer.added_tokens_decoder)
        model = ModelFactory.create(
            **config.model.to_dict(),
            pad_token_id=tokenizer.pad_token_id,
            num_added_tokens=num_added_tokens,
            device_map=device_map,
            resume=resume
        )
        trainer = TrainerFactory.create(config.trainer.name, model=model, dataloader=dataloader, config=config, resume=resume)
        trainer.train()
    dist.barrier()
    cleanup()


def train_single(config: List[GenericConfig], resume: bool):
    logger.info("Starting Experiment: %s", config.experiment.name)

    torch.manual_seed(int(config.experiment.seed))
    torch.cuda.manual_seed(int(config.experiment.seed))
    np.random.seed(int(config.experiment.seed))
    torch.cuda.empty_cache()

    device_map = config.experiment.device_map

    tokenizer = load_tokenizer(**config.tokenizer.to_dict())
    dataloader = DataLoaderFactory.create(**config.dataset.to_dict(), tokenizer=tokenizer)
    num_added_tokens = len(tokenizer.added_tokens_decoder)
    model = ModelFactory.create(
        **config.model.to_dict(),
        pad_token_id=tokenizer.pad_token_id,
        num_added_tokens=num_added_tokens,
        device_map=device_map,
        resume=resume
    )
    trainer = TrainerFactory.create(config.trainer.name, model=model, dataloader=dataloader, config=config, resume=resume)
    trainer.train()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
