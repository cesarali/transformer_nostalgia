import re
from pathlib import Path
from typing import Optional

from transformers import AddedToken, AutoTokenizer, PreTrainedTokenizerBase

from ...utils.helper import verify_str_arg
from . import ADataLoader, DataLoaderFactory

ALLOWED_TOKENIZERS = [
    None,
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
]
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# <s>[INST] {user_message_1} [/INST]
# <s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST]
# <s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]

# <s>[INST] {user_message_1} [/INST]
# <s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST]
# <s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]


def load_tokenizer(
    name: str, cache_dir: Optional[Path] = None, padding_side: Optional[str] = None, add_pad_token: bool = False, **kwargs
) -> PreTrainedTokenizerBase:
    """Helper funciton for loading a Huggingface tokenizer.

    Args:
        tokenizer_name (str): Name of the tokenizer.
        tokenizer_cache_dir (Path, optional): Path to the cache dir of the tokenizer. Defaults to None.
        padding_side (str, optional): Side of the sentence to pad. Defaults to None (takes the tokenizer as it is).
        add_pad_token (bool, optional): Add pad token. Llama2 models do not have pad token. Defaults to False.

    Returns:
        PreTrainedTokenizerBase: Instance of the chosen Huggingface tokenizer.
    """
    # verify_str_arg(name, "name", ALLOWED_TOKENIZERS)
    tokenizer = AutoTokenizer.from_pretrained(
        name, cache_dir=cache_dir, use_fast=False, token="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", **kwargs
    )
    if add_pad_token:
        tokenizer.add_special_tokens({"pad_token": AddedToken("<pad>", normalized=False)})

    if padding_side is not None:
        tokenizer.padding_side = verify_str_arg(padding_side, arg="padding_side", valid_values=["left", "right"])
    return tokenizer


class CommonsenseQADataLoader(ADataLoader):
    """Data loader for the CommonsenseQA dataset.


    Args:
        root_dir (Path | str): _description_
        device (_type_): _description_
        rank (int, optional): _description_. Defaults to 0.
        world_size (int, optional): _description_. Defaults to -1.

    References:
        - Talmor, Alon, et al. "Commonsenseqa: A question answering challenge targeting commonsense knowledge." arXiv preprint arXiv:1811.00937 (2018).
          URL: [Link to the paper](https://arxiv.org/pdf/1811.00937.pdf)

    """

    def __init__(self, **kwargs):
        super().__init__(ds_name="commonsense_qa", **kwargs)

    def _reformat_text(self, x: dict, is_test_split: bool) -> dict:
        labels = x["choices"]["label"]
        text = x["choices"]["text"]
        answer_key = x["answerKey"]
        answer = ""
        if answer_key != "":
            answer = f"({answer_key.lower()}) {text[labels.index(answer_key)]}"
        x["text_q"] = f'Q: {x["question"]}'
        x["text_q"] += " Answer Choices: " + "\n".join([f"({l.lower()}) {t}" for l, t in zip(labels, text)])
        if self.supervised and not is_test_split:
            x["text_a"] = f"\nA: The answer is {answer} </s>"
        if self.chat_style:
            x["text_q"] = f"{B_INST} {x['text_q']} {E_INST} "
        x["text_q"] = f"<s> {x['text_q']}"

        return x


DataLoaderFactory.register("commonsense_qa", CommonsenseQADataLoader)


class GSM8K(ADataLoader):
    def __init__(self, short_answer=True, **kwargs):
        self.short_answer = short_answer
        super().__init__(ds_name="gsm8k", **kwargs)

    def _reformat_text(self, x: dict, is_test_split: bool) -> dict:
        answer, label = re.split(r"\n#+\s", x["answer"])
        x["answerKey"] = int(label.replace(",", ""))

        x["text_q"] = f'Q: {x["question"]}'
        if self.supervised:
            if self.short_answer:
                x["text_a"] = f"\nA: The answer is {label}. </s>"
            else:
                x["text_a"] = f"\nA: {answer} The answer is {label}. </s>"
        if self.chat_style:
            x["text_q"] = f"{B_INST} {x['text_q']} {E_INST} "
        x["text_q"] = f"<s> {x['text_q']}"

        return x


DataLoaderFactory.register("gsm8k", GSM8K)


class Ipp50QA(ADataLoader):
    """Data loader for the CommonsenseQA dataset.


    Args:
        root_dir (Path | str): _description_
        device (_type_): _description_
        rank (int, optional): _description_. Defaults to 0.
        world_size (int, optional): _description_. Defaults to -1.

    References:
        - Talmor, Alon, et al. "Commonsenseqa: A question answering challenge targeting commonsense knowledge." arXiv preprint arXiv:1811.00937 (2018).
          URL: [Link to the paper](https://arxiv.org/pdf/1811.00937.pdf)

    """

    def __init__(self, **kwargs):
        super().__init__("cesarali/test_ipp50", **kwargs)

    def _reformat_text(self, x: dict, is_test_split: bool) -> dict:
        labels = ["A", "B", "C", "D", "E"]
        text = x["choices"]

        x["text_q"] = f'Q: Does the following statement: I {x["question"]} '
        x["text_q"] += "Indicate how you feel in which way? " + "\n".join([f"({l.lower()}) {t}" for l, t in zip(labels, text)])
        if self.chat_style:
            x["text_q"] = f"{B_INST} {x['text_q']} {E_INST}"
        x["text_q"] = f"<s> {x['text_q']}"
        return x


DataLoaderFactory.register("ipp50_qa", Ipp50QA)
