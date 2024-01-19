import collections
import functools
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from timeit import default_timer as timer
import datetime

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

        #self.activation_last_state = collections.defaultdict(list)
        for name, module in self.named_modules():
            module.register_forward_hook(functools.partial(self.save_activation, name))        
        self.activation_it = {}

        self.layers_to_save = []
        self.ids_to_save = []
        self._generate_new = False

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

    def save_activation(self, name : str, mod : nn.Module, inp : object, out, filter_black_list = []):
        if name in filter_black_list:
            return 
                
        if name not in self.layers_to_save:
            return        
        
        # skip all non-tensor objects
        if type(out) is not torch.Tensor:
            return

        for act_idx in range(out.size()[0]):
            if out.size()[1] > 1:
                # consider only the last activation for multiple inputs
                out = out[:,-1,:]
                out = out[:,None,:]
 
            self.save_state_to_disk(name,out[act_idx,:,:],self.ids_to_save[act_idx], self._generate_new)

    def set_generate_new_activations(self, generate_new : bool):
        self._generate_new = generate_new

    def save_state_to_disk(self, name : str, out, sub_directory : str, generate_new : bool, main_directory : str = "activations"):
        # create activation savings path iteratively
        main_directory_path = os.path.join(os.getcwd(),main_directory)
        if not os.path.exists(main_directory_path):
            # e.g. ./activations/
            os.mkdir(main_directory_path) 

        id_path = os.path.join(main_directory_path,sub_directory)
        if not os.path.exists(id_path):
            # e.g. ./activations/99af85081085e6228c6d78c95be01968/
            os.mkdir(id_path) 
        else: 
            if not generate_new:
                return

        act_id = self.activation_it.get(sub_directory)
        if act_id is None:
            self.activation_it[sub_directory] = 0

        

        act_path = os.path.join(id_path, str(self.activation_it.get(sub_directory)))        
        if not os.path.exists(act_path):
            # create subfolder per forward() call
            # e.g. ./activations/99af85081085e6228c6d78c95be01968/0/ 
            os.mkdir(act_path) 

        if os.path.exists(f"{act_path}/{name}.pt"):
            self.activation_it[sub_directory] += 1
            self.save_state_to_disk(name,out,sub_directory,main_directory)
       
        torch.save(out, f"{act_path}/{name}.pt")
    
    def set_layers_to_save(self, layer_names : List[str]):
        self.layers_to_save = layer_names

    def set_ids_to_save(self, id_names : List[str]):
        self.ids_to_save = id_names
    
    @staticmethod
    def load_state_from_disk(id : str, layer_name : str,  input_iteration : int, output_iteration : int, directory : str ="activations"):
        if input_iteration >= output_iteration or input_iteration < 0 or output_iteration < 0:
            print(f"Error while measuring activations. Please check input ({input_iteration}) and output ({output_iteration}) iteration.")
            return

        # check activation directory, session folder and iterations existence
        activation_path = os.path.join(os.getcwd(),directory)
        id_path = os.path.join(activation_path, id)
        input_it_path = os.path.join(id_path, str(input_iteration)) 
        output_it_path = os.path.join(id_path, str(output_iteration)) 

        if not os.path.exists(activation_path):
            raise ValueError(f"Error loading activations. Path not found: {activation_path}")
    
        if not os.path.exists(id_path):
            raise ValueError(f"Error loading activations. Path not found: {id_path}")
        
        if not os.path.exists(input_it_path):
            raise ValueError(f"Error loading activations. Path not found: {input_it_path}")
        
        if not os.path.exists(output_it_path):
            raise ValueError(f"Error loading activations. Path not found: {output_it_path}")

        # check if layers exist
        input_layer = os.path.join(input_it_path, f"{layer_name}.pt")
        output_layer = os.path.join(output_it_path, f"{layer_name}.pt") 

        if not os.path.exists(input_layer):
            raise ValueError(f"Error loading activations. File not found: {input_layer}")
        
        if not os.path.exists(output_layer):
            raise ValueError(f"Error loading activations. File not found: {output_layer}")

        input_layer_activation = torch.load(input_layer)
        output_layer_activation = torch.load(output_layer)

        # check if dimensionalities are equal for both iterations:
        input_dim = input_layer_activation.size()
        output_dim = output_layer_activation.size()

        if input_dim != output_dim:
            raise ValueError(f"Error loading activations. Dimensions do not match: input dimension ({input_dim}) is not equal to output dimension ({output_dim})!")
        
        return input_layer_activation, output_layer_activation

ModelFactory.register("LLMCausal", LLMCausal)
