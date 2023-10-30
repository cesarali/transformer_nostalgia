import functools
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import PeftModel
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
from peft.utils import get_peft_model_state_dict
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block,
    GPT2Config,
    GPT2LMHeadModel,
)
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
)

from ..trainers.mixed_precision import is_bfloat_supported
from ..utils.logging import RankLoggerAdapter
from .utils import add_peft_adapter, freeze_transformer_layers


def is_distributed() -> bool:
    return dist.is_initialized()


class AModel(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    @abstractmethod
    def new_stats(self) -> Dict:
        """
        Create dictionary where it will hold the results (_loss_ and _metrics_) after each training step.
        :return:
        """
        raise NotImplementedError("The new_stats method is not implemented in your class!")

    @abstractmethod
    def loss(self, *inputs) -> Dict:
        raise NotImplementedError("The loss method is not implemented in your class!")

    @abstractmethod
    def metric(self, y: Any, y_target: Any) -> Dict:
        raise NotImplementedError("The metric method is not implemented in your class!")

    @abstractmethod
    def train_step(self, batch: dict, optimizers: dict = None, schedulers: dict = None, gradient_accumulation_steps: int = 1) -> Dict:
        raise NotImplementedError("The train_step method is not implemented in your class!")

    @abstractmethod
    def validate_step(self, batch: Any) -> Dict:
        raise NotImplementedError("The validate_step method is not implemented in your class!")

    def validate_epoch(self, dataloader: Any = None, epoch: int = 1) -> Dict:
        """
        This is performed only once every
            save_after_epoch: 10
            and usually accounts for global validation metrics, like the evaluation of the full graph etc
        :return:
        """
        return {}

    def fsdp_wrap_policy(self):
        return None

    @property
    def device(self):
        if is_distributed():
            return int(os.environ["LOCAL_RANK"])
        return next(self.parameters()).device

    @property
    def rank(self) -> int:
        if is_distributed():
            return int(os.environ["RANK"])
        return 0


class ModelFactory:
    model_types = {}

    @classmethod
    def register(cls, model_type: str, model_class: AModel):
        cls.model_types[model_type] = model_class

    @classmethod
    def create(cls, name: str, **kwargs) -> AModel:
        model_class = cls.model_types.get(name)
        if model_class:
            return model_class(**kwargs)
        else:
            raise ValueError("Invalid model type")


class LLM(AModel):
    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_bf16: bool = False,
        device_map: Optional[str] = None,
        resume: bool = False,
        peft: Optional[dict] = None,
    ):
        super(LLM, self).__init__()

        self._device_map = device_map
        self._torch_dtype = None
        self._quantization_config = None
        self.resume = resume
        self.peft = peft
        if load_in_8bit and load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif load_in_8bit or load_in_4bit:
            self._quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
            self._device_map = "auto"
        if use_bf16 and is_bfloat_supported:
            self._torch_dtype = torch.float16

    def forward(self, batch, schedulers: Optional[dict] = None, step: Optional[int] = None):
        """
        Forward step of the  language model.

        Parameters
        ----------
        input (Tensor) of shape [B, T, D]
        z (Tensor) of shape [B, D'] representing global dynamic state

        Returns
        -------
        (logits, hidden_state)
        Notation. B: batch size; T: seq len (== fix_len); D: hidden dimension
        """

        out = self.backbone(**batch)
        return {"losses": {"loss": out.loss, "nll-loss": out.loss}, "logits": out.logits}

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        return_full_text: bool = False,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        """
        Forward step of the  language model.

        Parameters
        ----------
        input (Tensor) of shape [B, T, D]
        z (Tensor) of shape [B, D'] representing global dynamic state

        Returns
        -------
        (logits, hidden_state)
        Notation. B: batch size; T: seq len (== fix_len); D: hidden dimension
        """

        out = self.backbone.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if tokenizer is not None:
            out = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if not return_full_text:
                inp_txt = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                out = list(map(lambda x: x[1][len(x[0]) :], zip(inp_txt, out)))
                return out
        return out

    def metric(self, y: Any, y_target: Any, seq_len=None):
        """
        returns a dictionary with metrics
        """

        return {}

    def train_step(self, batch: Any, optimizers: Any = None, schedulers: Any = None, gradient_accumulation_steps: int = 1):
        return {}

    def validate_step(self, batch: Any):
        return {}

    def new_metric_stats(self) -> Dict:
        stats = dict()
        return stats

    def fsdp_activation_check_fn(self):
        if isinstance(self.backbone, GPT2LMHeadModel):
            return lambda submodule: isinstance(submodule, GPT2Block)
        elif isinstance(self.backbone, LlamaForCausalLM):
            return lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        else:
            raise ValueError("Activation checkpoint is not defined for this type of models!")

    def get_transformer_layers(self):
        if isinstance(self.config, GPT2Config):
            return {GPT2Block}
        elif isinstance(self.config, LlamaConfig):
            return {LlamaDecoderLayer}
        else:
            raise ValueError("Wrapping policy is not defined for this type of models!")

    def fsdp_wrap_policy(self):
        transformer_layers = self.get_transformer_layers()
        return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layers)

    def fsdp_peft_wrap_policy(self):
        def lambda_policy_fn(module):
            return len(list(module.named_children())) == 0 and getattr(module, "weight", None) is not None and module.weight.requires_grad

        transformer_layers = self.get_transformer_layers()
        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        transformer_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={PrefixEncoder, PromptEncoder, PromptEmbedding} | transformer_layers
        )
        auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
        return auto_wrap_policy
        # return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layers)

    def get_fsdp_policy(self, min_num_params: int):
        try:
            if self.is_peft():
                wrap_policy = self.fsdp_peft_wrap_policy()
            else:
                wrap_policy = self.fsdp_wrap_policy()
        except ValueError:
            self.logger.warning("The model does not have custom wrapping policy. Size based auto policy will be used!")
            wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        return wrap_policy

    def new_stats(self) -> Dict:
        stats = dict()
        return stats

    def is_peft(self) -> bool:
        return self.peft is not None and self.peft["method"] is not None


class LLMCausal(LLM):
    def __init__(
        self,
        backbone: str,
        backbone_path: Optional[Path] = None,
        pad_token_id: Optional[int] = None,
        num_added_tokens: Optional[int] = None,
        freeze_first_n_layers: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self.num_added_tokens = num_added_tokens
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone,
            cache_dir=backbone_path,
            device_map=self._device_map,
            torch_dtype=self._torch_dtype,
            quantization_config=self._quantization_config,
            token="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",
            trust_remote_code=True,
        )

        self.backbone.config.tie_word_embedding = False
        if pad_token_id is not None and (self.backbone.config.pad_token_id == 0 or self.backbone.config.pad_token_id is None):
            self.backbone.config.pad_token_id = pad_token_id
            if num_added_tokens > 0:
                self.backbone.resize_token_embeddings(self.backbone.config.vocab_size + num_added_tokens)
            if hasattr(self.backbone, "transformer"):
                self.backbone.transformer.wte.padding_idx = pad_token_id
                self.backbone.transformer.wte._fill_padding_idx_with_zero()
            elif hasattr(self.backbone, "model"):
                self.backbone.model.embed_tokens.padding_idx = pad_token_id
                self.backbone.model.embed_tokens._fill_padding_idx_with_zero()
            else:
                raise TypeError("Unknown Language Model Type. Padding token cannot be added to the lanbguage embedding layer!")
        if self.is_peft() and (not self.resume or self.rank != 0):
            add_peft_adapter(self.backbone, self.peft)
        if not self.is_peft() and freeze_first_n_layers is not None:
            freeze_transformer_layers(self.backbone, freeze_first_n_layers)

    def loss(self, logits, labels):
        """
        returns the cross_entropy for the language model
        Notation. B: batch size; T: seq len (== fix_len)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss}

    @property
    def peft_config(self):
        return self.backbone.peft_config["default"]

    @property
    def config(self):
        return self.backbone.config

    def save_pretrained(self, output_dir: Path):
        return self.backbone.save_pretrained(output_dir)

    def get_peft_state_dict(self):
        return {"backbone": get_peft_model_state_dict(self.backbone)}

    def load_peft_pretrained_model(self, path: Path):
        backbone = PeftModel.from_pretrained(self.backbone, path, is_trainable=True)
        self.backbone = backbone


ModelFactory.register("LLMCausal", LLMCausal)
