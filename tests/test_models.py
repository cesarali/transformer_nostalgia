# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import pytest
import torch

from nostalgia import test_data_path
from nostalgia.data import load_tokenizer
from nostalgia.data.dataloaders import DataLoaderFactory
from nostalgia.models import ModelFactory


class TestLLM:
    conf_path = test_data_path / "config" / "gpt2_commonsense_qa.yaml"

    def test_init_llm(self):
        backbone = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        num_added_tokens = len(tokenizer.added_tokens_decoder)
        model = ModelFactory.create(
            name="LLMCausal",
            pad_token_id=tokenizer.pad_token_id,
            num_added_tokens=num_added_tokens,
            backbone=backbone,
            peft={"method": None},
        )
        assert model is not None
        assert model.device == torch.device("cpu")
        del model
        model = ModelFactory.create(
            name="LLMCausal",
            pad_token_id=tokenizer.pad_token_id,
            num_added_tokens=num_added_tokens,
            backbone=backbone,
            device_map="cuda",
            peft={"method": None},
        )
        assert model.device == torch.device("cuda:0")
        del model
        model = ModelFactory.create(
            name="LLMCausal",
            pad_token_id=tokenizer.pad_token_id,
            num_added_tokens=num_added_tokens,
            backbone=backbone,
            device_map="auto",
            peft={"method": None},
        )
        assert model.device == torch.device("cuda:0")

    @pytest.mark.skip()
    def test_model_factory_seq2seq(self):
        backbone = "MBZUAI/LaMini-T5-61M"
        tokenizer = load_tokenizer(backbone)
        assert tokenizer is not None
        model_params = {"backbone": backbone}
        model = ModelFactory.create("LLMSeq2Seq", **model_params)
        assert model is not None

    def test_model_factory_causal(self):
        backbone = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = load_tokenizer(backbone, add_pad_token=True)
        num_added_tokens = len(tokenizer.added_tokens_decoder)
        assert tokenizer is not None
        model_params = {"backbone": backbone, "device_map": "cuda", "peft": {"method": None}}
        num_added_tokens = len(tokenizer.added_tokens_decoder)
        model = ModelFactory.create("LLMCausal", **model_params, pad_token_id=tokenizer.pad_token_id, num_added_tokens=num_added_tokens)
        assert model is not None
        dataloader = DataLoaderFactory.create(
            "commonsense_qa",
            batch_size=1,
            output_fields=["input_ids", "attention_mask", "labels"],
            tokenizer=tokenizer,
            supervised=True,
            max_padding_length=150,
            target_type="INSTRUCTION_FINTUNE",
            force_download=True,
        )
        minibatch = next(iter(dataloader.train_it))
        model.train()
        for k, v in minibatch.items():
            minibatch[k] = v.to("cuda")
        losses = model(batch=minibatch)["losses"]
        assert isinstance(losses, dict)
