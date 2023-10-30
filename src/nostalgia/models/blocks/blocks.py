import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Config
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer

from ...trainers.mixed_precision import is_bfloat_supported
from ...trainers.utils import is_distributed
from ...utils.logging import RankLoggerAdapter
from ..utils import add_peft_adapter, freeze_transformer_layers


class Block(nn.Module):
    def __init__(self, rw_length: int = 0, n_symbols: int = 0, resume: bool = False, **kwargs):
        super(Block, self).__init__()
        self.rw_length = rw_length
        self.n_symbols = n_symbols

        self.resume = resume

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

    def param_init(self):
        """
        Parameters initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias.data is not None:
                    nn.init.zeros_(module.bias)


class HFBlock(Block):
    """
    Hugging Face Transformer-based neural network block.

    Args:
        backbone (str): Name of the Hugging Face transformer backbone.
        pad_token_id (int): Padding token ID.
        num_added_tokens (int): Number of additional tokens.
        load_in_8bit (bool): Load the model with 8-bit quantization.
        load_in_4bit (bool): Load the model with 4-bit quantization.
        use_bf16 (bool): Use bfloat16 data type for model weights.
        **kwargs: Additional keyword arguments for the base class.

    Attributes:
        _torch_dtype: Torch data type for model weights (None if not used).
        _quantization_config: Quantization configuration (None if not used).
        backbone: Hugging Face transformer backbone model.
        conf: AutoConfig instance.
    """

    def __init__(
        self,
        backbone: str | PreTrainedModel,
        backbone_path: Optional[Path] = None,
        pad_token_id: Optional[int] = None,
        num_added_tokens: Optional[int] = 0,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_bf16: bool = False,
        device_map: Optional[str] = None,
        peft: Optional[dict] = None,
        freeze_first_n_layers: Optional[int] = None,
        freeze_backbone: Optional[bool] = False,
        adapter_name: Optional[str] = None,
        symbol_dim: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self._device_map = device_map
        self._torch_dtype = None
        self._quantization_config = None
        self.is_peft = peft is not None and peft["method"] is not None
        self.backbone = None
        if isinstance(backbone, str):
            self._load_backbone(backbone, backbone_path, load_in_8bit, load_in_4bit, use_bf16, pad_token_id, num_added_tokens, symbol_dim)
        else:
            self.config = backbone.config
        if self.is_peft and (not self.resume or self.rank != 0):
            add_peft_adapter(backbone if self.backbone is None else self.backbone, peft, adapter_name)

        assert not (freeze_backbone and freeze_first_n_layers is not None), self.logger.error(
            "Both `freeze_backbone` and `freeze_first_n_layers` are set at the same time!"
        )
        if not self.is_peft and freeze_backbone:
            for params in self.backbone.parameters():
                params.require_grad = False

        if not self.is_peft and freeze_first_n_layers is not None:
            freeze_transformer_layers(self.backbone, freeze_first_n_layers)

    def _load_backbone(
        self,
        backbone,
        backbone_path,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_bf16: bool = False,
        pad_token_id: Optional[int] = None,
        num_added_tokens: Optional[int] = 0,
        symbol_dim: Optional[int] = 0,
    ):
        if load_in_8bit and load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif load_in_8bit or load_in_4bit:
            self._quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
            self._device_map = "auto"
        if use_bf16 and is_bfloat_supported:
            self._torch_dtype = torch.float16

        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone,
            device_map=self._device_map,
            torch_dtype=self._torch_dtype,
            quantization_config=self._quantization_config,
            cache_dir=backbone_path,
            token="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",
        )
        self.config = self.backbone.config
        if num_added_tokens > 0 or pad_token_id is not None:
            self.backbone.resize_token_embeddings(self.config.vocab_size + num_added_tokens, pad_to_multiple_of=256)
        if pad_token_id is not None and (self.config.pad_token_id == 0 or self.config.pad_token_id is None):
            if hasattr(self.backbone, "transformer"):
                self.backbone.transformer.wte.padding_idx = pad_token_id
                self.backbone.transformer.wte._fill_padding_idx_with_zero()
            elif hasattr(self.backbone, "model"):
                self.backbone.model.embed_tokens.padding_idx = pad_token_id
                self.backbone.model.embed_tokens._fill_padding_idx_with_zero()
            elif hasattr(self.backbone, "embeddings"):
                self.backbone.embeddings.word_embeddings.padding_idx = pad_token_id
                self.backbone.embeddings.word_embeddings._fill_padding_idx_with_zero()
            else:
                raise TypeError("Unknown Language Model Type. Padding token cannot be added to the lanbguage embedding layer!")

    def get_transformer_layers(self):
        if isinstance(self.config, GPT2Config):  # EncoderModelA, Decoder
            return {GPT2Block}
        elif isinstance(self.config, LlamaConfig):
            return {LlamaDecoderLayer}
        else:
            raise ValueError("Wrapping policy is not defined for this type of models!")

    def __str__(self):
        """
        Get a string representation of the HFBlock instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"HFBlock(backbone={self.backbone.config.architectures[0]}, dtype={self._torch_dtype}, quantization={self._quantization_config})"

    def forward(self, **kwargs):
        return self.backbone(**kwargs)

