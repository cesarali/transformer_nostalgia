import logging
from abc import ABC
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from datasets import (
    DatasetDict,
    DownloadMode,
    get_dataset_split_names,
    load_dataset,
    load_dataset_builder,
)
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerBase

from ...trainers.utils import is_distributed
from ...utils.helper import load_prompting_text, verify_int_arg, verify_str_arg
from ...utils.logging import RankLoggerAdapter

DistributedSampler = torch.utils.data.distributed.DistributedSampler


class TargetType(Enum):
    SEQ2SEQ = auto()
    INSTRUCTION_FINTUNE = auto()


class ADataLoader(ABC):
    def __init__(
        self,
        ds_name: str,
        root_dir: Optional[Path | str] = None,
        split: Optional[str] = None,
        prompt_path: Optional[Path | str] = None,
        wrap_prompt_as: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_padding_length: Optional[int] = 2048,
        output_fields: Optional[List[str]] = None,
        supervised: Optional[bool] = False,
        force_download: Optional[bool] = False,
        chat_style: Optional[bool] = False,
        target_type: Optional[str] = "SEQ2SEQ",
        **kwargs,
    ):
        self.batch_size = kwargs.pop("batch_size")
        self.test_batch_size = kwargs.pop("test_batch_size", self.batch_size)
        self.path_to_vectors = kwargs.pop("path_to_vectors", None)
        self.emb_dim = kwargs.pop("emb_dim", None)
        self.voc_size = kwargs.pop("voc_size", None)
        self.min_freq = kwargs.pop("min_freq", 1)
        self._fix_length = kwargs.pop("fix_len", None)
        self.min_len = kwargs.pop("min_len", None)
        self.max_len = kwargs.pop("max_len", None)
        self.lower = kwargs.pop("lower", False)
        self.punctuation = kwargs.pop("punctuation", True)
        self.dataset_kwargs = kwargs
        self.ds_name = ds_name
        self.iter = {}

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.root_dir = root_dir if root_dir is None else Path(root_dir)
        self.supervised = supervised
        self.tokenizer = tokenizer
        self.chat_style = chat_style
        self.target_type = TargetType[target_type]
        if self.tokenizer is not None:
            self.tokenizer.add_special_tokens = True
            self.tokenizer.add_bos_token = False

        self.split = verify_str_arg(split, arg="split", valid_values=get_dataset_split_names(ds_name, "default") + [None])
        self.max_padding_length = verify_int_arg(max_padding_length, arg="max_padding_length", min_value=0, max_value=2048)
        self.wrap_prompt_as = verify_str_arg(wrap_prompt_as, arg="wrap_prompt_as", valid_values=[None, "instruction", "context"])

        if self.split is not None:
            self.dataset = DatasetDict(
                {
                    self.split: load_dataset(
                        ds_name,
                        "default",
                        cache_dir=self.root_dir,
                        download_mode=DownloadMode.FORCE_REDOWNLOAD if force_download else None,
                        split=self.split,
                        token="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",
                    )
                }
            )
        else:
            self.dataset = DatasetDict(
                [
                    (
                        name,
                        load_dataset(
                            ds_name,
                            "main",
                            cache_dir=self.root_dir,
                            download_mode=DownloadMode.FORCE_REDOWNLOAD if force_download else None,
                            split=name,
                            token="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",
                        ),
                    )
                    for name in get_dataset_split_names(ds_name, "main")
                ]
            )

        for _split, _dataset in self.dataset.items():
            self.dataset[_split] = _dataset.map(partial(self._reformat_text, is_test_split=_split == "test"))

        if prompt_path is not None:
            self.__prefix_with_text(prompt_path)

        if self.tokenizer is not None:
            for _split, _dataset in self.dataset.items():
                self.dataset[_split] = _dataset.map(partial(self._tokenize_fn, is_test_split=_split == "test"))
                _out_fields = None
                if output_fields is not None:
                    _out_fields = list(set(self.dataset[_split].features.keys()) & set(output_fields))
                self.dataset[_split].set_format(type="torch", columns=_out_fields)
        self._init_dataloaders(self.dataset)

    def _init_dataloaders(self, dataset):
        for n, d in dataset.items():
            sampler = None
            if is_distributed():
                sampler = DistributedSampler(d, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=n == "train")
            batch_size = self.batch_size
            if n != "train":
                batch_size = self.test_batch_size
            self.iter[n] = DataLoader(
                d,
                drop_last=False,
                sampler=sampler,
                shuffle=sampler is None,
                batch_size=batch_size,
                # collate_fn=default_data_collator,
                **self.dataset_kwargs,
            )

    def _tokenize_fn(self, x: dict, is_test_split: bool) -> dict:
        if self.supervised and not is_test_split:
            tokenized = self.tokenizer(
                x["text_q"],
                x["text_a"],
                return_token_type_ids=True,
                truncation="only_second",
                padding="max_length",
                max_length=self.max_padding_length,
            )

            token_type_ids = np.asarray(tokenized["token_type_ids"])
            labels = np.asarray(tokenized["input_ids"].copy())
            if self.target_type == TargetType.INSTRUCTION_FINTUNE:
                labels[token_type_ids == 0] = -100
            elif self.target_type == TargetType.SEQ2SEQ:
                labels[labels == self.tokenizer.pad_token_id] = -100
            else:
                raise ValueError(f"Invalid target type {self.target_type}")
            tokenized["labels"] = labels.tolist()

            return tokenized
        else:
            return self.tokenizer(x["text_q"], return_token_type_ids=True, padding="max_length", max_length=self.max_padding_length)

    def __prefix_with_text(self, prompt_path):
        self.prompting_input = load_prompting_text(prompt_path)

        def add_prompt(x):
            if self.wrap_prompt_as == "instruction":
                x["text_q"] = "<s>[INST] " + self.prompting_input + "\n" + x["text_q"] + " [/INST]"
            elif self.wrap_prompt_as == "context":
                x["text_q"] = "<s>[INST] <<SYS>>\n" + self.prompting_input + "\n<</SYS>>\n\n" + x["text_q"] + " [/INST]"
            else:
                x["text_q"] = self.prompting_input + "\n" + x["text_q"]
            return x

        self.dataset = self.dataset.map(add_prompt)

    @property
    def number_add_tokens(self) -> int:
        """Returns the number of added new tokens to the tokenizer.

        Returns:
            int: number of new tokens
        """
        return 1

    def __str__(self) -> str:
        ds_info = load_dataset_builder(self.ds_name)
        return f"{ds_info.info.description}\n{ds_info.info.features}"

    @property
    def train(self):
        return self.iter["train"].dataset

    @property
    def train_it(self) -> DataLoader:
        return self.iter["train"]

    @property
    def validation(self):
        return self.iter["validation"].dataset

    @property
    def validation_it(self) -> DataLoader:
        return self.iter["validation"]

    @property
    def test(self):
        return self.iter["test"].dataset

    @property
    def test_it(self) -> DataLoader:
        return self.iter["test"]

    @property
    def n_train_batches(self):
        return len(self.train_it)

    @property
    def n_validation_batches(self):
        return len(self.validation_it)

    @property
    def n_test_batches(self):
        return len(self.test_it)

    @property
    def train_set_size(self):
        return len(self.train)

    @property
    def validation_set_size(self):
        return len(self.validation)

    @property
    def test_set_size(self):
        return len(self.test)


class DataLoaderFactory:
    """Dataloader factory class."""

    object_types = {}

    @classmethod
    def register(cls, object_type: str, object_class: ADataLoader) -> None:
        """Register new dataloader type to the factory.

        Args:
            object_type (str): name of the object
            object_class (ADataLoader): class that is registered
        """
        cls.object_types[object_type] = object_class

    @classmethod
    def create(cls, name: str, **kwargs) -> ADataLoader:
        """Create new dataloader object.

        Args:
            object_type (str): name of the object type that is created

        Raises:
            ValueError: if the object type is not registered

        Returns:
            ADataLoader: new instance of the dataloader object
        """
        object_class = cls.object_types.get(name)
        if object_class:
            return object_class(**kwargs)
        else:
            raise ValueError("Invalid object type!")
