# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import torch

from nostalgia import test_data_path
from nostalgia.models import ModelFactory
from nostalgia.utils.helper import (
    GenericConfig,
    create_optimizers,
    load_prompting_text,
    load_yaml,
)

TRAIN_CONF = test_data_path / "config" / "gpt2_commonsense_qa.yaml"


def test_load_prompting_text():
    PATH = test_data_path / "prompts" / "commonsenseQA_TOT.txt"
    prompt = load_prompting_text(PATH)
    assert len(prompt) == 2_130


def test_load_prompting_yaml():
    config = load_yaml(TRAIN_CONF)

    assert isinstance(config, dict)

    config = load_yaml(TRAIN_CONF, True)

    assert isinstance(config, GenericConfig)
    assert config.experiment.name == "test"
    assert isinstance(config.model, GenericConfig)
    print(config.model.__dict__)


def test_create_optimizer():
    config = load_yaml(TRAIN_CONF, True)
    print(config.optimizers)
    model = ModelFactory.create(**config.model.to_dict())
    optimizers = create_optimizers(model, config.optimizers)
    print(optimizers)
    assert optimizers is not None
    assert isinstance(optimizers["optimizer_d"]["opt"], torch.optim.AdamW)
