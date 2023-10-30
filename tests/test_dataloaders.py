# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
import pytest
from nostalgia import test_data_path
from nostalgia.data import DataLoaderFactory, load_tokenizer
from nostalgia.utils.helper import load_prompting_text


class TestGSM8K:
    def test_load(self):
        dataset = DataLoaderFactory.create("gsm8k", batch_size=32, supervised=True)
        assert len(dataset.train) == 7_473
        assert len(dataset.test) == 1_319

        assert (
            dataset.train[0]["text_q"]
            == "<s> Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
        )
        assert dataset.train[0]["text_a"] == "\nA: The answer is 72. </s>"
        assert "text_a" in dataset.train.features

    def test_load_long_answers(self):
        dataset = DataLoaderFactory.create("gsm8k", batch_size=32, supervised=True, short_answer=False, force_download=True)
        assert len(dataset.train) == 7_473
        assert len(dataset.test) == 1_319

        assert (
            dataset.train[0]["text_q"]
            == "<s> Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
        )
        assert (
            dataset.train[0]["text_a"]
            == "\nA: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. The answer is 72. </s>"
        )
        assert "text_a" in dataset.train.features

    def test_split(self):
        dataset = DataLoaderFactory.create("gsm8k", batch_size=32, test_batch_size=8, supervised=True, split="test")

        assert len(dataset.test) == 1_319

    def test_load_chat(self):
        dataset = DataLoaderFactory.create("gsm8k", batch_size=32, supervised=True, chat_style=True, force_download=True)
        assert len(dataset.train) == 7_473
        assert len(dataset.test) == 1_319

        assert (
            dataset.train[0]["text_q"]
            == "<s> [INST] Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST] "
        )
        assert dataset.train[0]["text_a"] == "\nA: The answer is 72. </s>"
        assert "text_a" in dataset.train.features

    def test_load_tokenized(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create("gsm8k", batch_size=32, tokenizer=tokenizer, supervised=True, max_padding_length=150)
        assert len(dataset.train) == 7_473
        assert len(dataset.test) == 1_319

        assert dataset.train[0]["input_ids"][0] == 1

    def test_target_type(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create(
            "gsm8k", batch_size=32, tokenizer=tokenizer, supervised=True, max_padding_length=150, target_type="INSTRUCTION_FINTUNE"
        )
        assert len(dataset.train) == 7_473
        assert len(dataset.test) == 1_319

        assert dataset.train[0]["labels"][0] == -100

        dataset = DataLoaderFactory.create(
            "gsm8k",
            batch_size=32,
            tokenizer=tokenizer,
            supervised=True,
            max_padding_length=150,
            target_type="SEQ2SEQ",
            force_download=True,
        )
        assert dataset.train[0]["labels"][0] == 1
        assert dataset.train[0]["labels"][-1] == -100

    def test_load_tokenized_chat(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create(
            "gsm8k", batch_size=32, tokenizer=tokenizer, supervised=True, max_padding_length=150, chat_style=True
        )
        assert len(dataset.train) == 7_473
        assert len(dataset.test) == 1_319

        assert dataset.train[0]["input_ids"][0] == 1
        assert dataset.train[0]["input_ids"][2] == 25580


class TestCSQA:
    def test_split_commonsenseqa(self):
        dataset = DataLoaderFactory.create("commonsense_qa", batch_size=32, test_batch_size=8, supervised=True, split="validation")

        assert len(dataset.validation) == 1_221

    def test_load_commonsenseqa(self):
        dataset = DataLoaderFactory.create("commonsense_qa", batch_size=32, supervised=True)
        assert len(dataset.train) == 9_741
        assert len(dataset.validation) == 1_221

        assert (
            dataset.train[0]["text_q"]
            == "<s> Q: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Answer Choices: (a) ignore\n(b) enforce\n(c) authoritarian\n(d) yell at\n(e) avoid"
        )
        assert dataset.train[0]["text_a"] == "\nA: The answer is (a) ignore </s>"
        assert "text_a" in dataset.train.features

    def test_load_commonsenseqa_chat(self):
        dataset = DataLoaderFactory.create("commonsense_qa", batch_size=32, supervised=True, chat_style=True, force_download=True)
        assert len(dataset.train) == 9_741
        assert len(dataset.validation) == 1_221

        assert (
            dataset.train[0]["text_q"]
            == "<s> [INST] Q: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Answer Choices: (a) ignore\n(b) enforce\n(c) authoritarian\n(d) yell at\n(e) avoid [/INST] "
        )
        assert dataset.train[0]["text_a"] == "\nA: The answer is (a) ignore </s>"
        assert "text_a" in dataset.train.features

    def test_load_tokenized_commonsenseqa(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create("commonsense_qa", batch_size=32, tokenizer=tokenizer, supervised=True, max_padding_length=150)
        assert len(dataset.train) == 9_741
        assert len(dataset.validation) == 1_221

        assert dataset.train[0]["input_ids"][0] == 1

    def test_target_type_commonsenseqa(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create(
            "commonsense_qa", batch_size=32, tokenizer=tokenizer, supervised=True, max_padding_length=150, target_type="INSTRUCTION_FINTUNE"
        )
        assert len(dataset.train) == 9_741
        assert len(dataset.validation) == 1_221

        assert dataset.train[0]["labels"][0] == -100

        dataset = DataLoaderFactory.create(
            "commonsense_qa",
            batch_size=32,
            tokenizer=tokenizer,
            supervised=True,
            max_padding_length=150,
            target_type="SEQ2SEQ",
            force_download=True,
        )
        assert dataset.train[0]["labels"][0] == 1
        assert dataset.train[0]["labels"][-1] == -100

    def test_load_tokenized_commonsenseqa_chat(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create(
            "commonsense_qa", batch_size=32, tokenizer=tokenizer, supervised=True, max_padding_length=150, chat_style=True
        )
        assert len(dataset.train) == 9_741
        assert len(dataset.validation) == 1_221

        assert dataset.train[0]["input_ids"][0] == 1
        assert dataset.train[0]["input_ids"][2] == 25580

    @pytest.mark.skip()
    def test_load_prompting_commonsenseqa(self):
        PROMPTS_PATH = test_data_path / "prompts" / "commonsenseQA_TOT.txt"
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create("commonsense_qa", batch_size=32, tokenizer=tokenizer, prompt_path=PROMPTS_PATH)

        PROMPT_TEXT = load_prompting_text(PROMPTS_PATH)

        assert (
            dataset.train[0]["text_q"]
            == PROMPT_TEXT
            + "\nQ: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Answer Choices: (a) ignore\n(b) enforce\n(c) authoritarian\n(d) yell at\n(e) avoid"
        )

    @pytest.mark.skip()
    def test_load_prompting_as_instruction_commonsenseqa(self):
        PROMPTS_PATH = test_data_path / "prompts" / "commonsenseQA_SP.txt"
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create(
            "commonsense_qa", batch_size=32, tokenizer=tokenizer, prompt_path=PROMPTS_PATH, wrap_prompt_as="instruction"
        )
        print(dataset.train)
        print(dataset.train[0])
        PROMPT_TEXT = load_prompting_text(PROMPTS_PATH)

        assert (
            dataset.train[0]["text_q"]
            == "<s>[INST] "
            + PROMPT_TEXT
            + "\nQ: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Answer Choices: (a) ignore\n(b) enforce\n(c) authoritarian\n(d) yell at\n(e) avoid"
            + " [/INST]"
        )

        dataset = DataLoaderFactory.create(
            "commonsense_qa", batch_size=32, tokenizer=tokenizer, prompt_path=PROMPTS_PATH, wrap_prompt_as="context"
        )
        assert (
            dataset.train[0]["text_q"]
            == "<s>[INST] <<SYS>>\n"
            + PROMPT_TEXT
            + "\n<</SYS>>"
            + "\n\nQ: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Answer Choices: (a) ignore\n(b) enforce\n(c) authoritarian\n(d) yell at\n(e) avoid"
            + " [/INST]"
        )


class TestIpp50QA:
    def test_split(self):
        dataset = DataLoaderFactory.create("ipp50_qa", batch_size=32, test_batch_size=8, supervised=True, split="train")

        assert len(dataset.train) == 50

    def test_load(self):
        dataset = DataLoaderFactory.create("ipp50_qa", batch_size=32, supervised=True, force_download=True)
        assert len(dataset.train) == 50

        assert (
            dataset.train[0]["text_q"]
            == "<s> Q: Does the following statement: I Am the life of the party. Indicate how you feel in which way? (a) Very Inaccurate\n(b) Moderately Inaccurate\n(c) Neither Accurate Nor Inaccurate\n(d) Moderately Accurate\n(e) Very Accurate </s>"
        )

    def test_load_chat(self):
        dataset = DataLoaderFactory.create("ipp50_qa", batch_size=32, force_download=True, chat_style=True)
        assert len(dataset.train) == 50

        assert (
            dataset.train[0]["text_q"]
            == "<s> [INST] Q: Does the following statement: I Am the life of the party. Indicate how you feel in which way? (a) Very Inaccurate\n(b) Moderately Inaccurate\n(c) Neither Accurate Nor Inaccurate\n(d) Moderately Accurate\n(e) Very Accurate [/INST] </s>"
        )

    def test_load_tokenized(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create("ipp50_qa", batch_size=32, tokenizer=tokenizer, max_padding_length=150)
        assert len(dataset.train) == 50

        assert dataset.train[0]["input_ids"][0] == 1

    def test_target_type(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create(
            "ipp50_qa", batch_size=32, tokenizer=tokenizer, force_download=True, max_padding_length=150, target_type="INSTRUCTION_FINTUNE"
        )
        assert len(dataset.train) == 50

        dataset = DataLoaderFactory.create(
            "ipp50_qa", batch_size=32, tokenizer=tokenizer, max_padding_length=150, target_type="SEQ2SEQ", force_download=True
        )

    def test_load_tokenized_chat(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create(
            "ipp50_qa", batch_size=32, tokenizer=tokenizer,  max_padding_length=150, chat_style=True
        )
        assert len(dataset.train) == 50

        assert dataset.train[0]["input_ids"][0] == 1
        assert dataset.train[0]["input_ids"][2] == 25580

    @pytest.mark.skip()
    def test_load_prompting_commonsenseqa(self):
        PROMPTS_PATH = test_data_path / "prompts" / "commonsenseQA_TOT.txt"
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create("commonsense_qa", batch_size=32, tokenizer=tokenizer, prompt_path=PROMPTS_PATH)

        PROMPT_TEXT = load_prompting_text(PROMPTS_PATH)

        assert (
            dataset.train[0]["text_q"]
            == PROMPT_TEXT
            + "\nQ: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Answer Choices: (a) ignore\n(b) enforce\n(c) authoritarian\n(d) yell at\n(e) avoid"
        )

    @pytest.mark.skip()
    def test_load_prompting_as_instruction_commonsenseqa(self):
        PROMPTS_PATH = test_data_path / "prompts" / "commonsenseQA_SP.txt"
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf", add_pad_token=True)
        dataset = DataLoaderFactory.create(
            "commonsense_qa", batch_size=32, tokenizer=tokenizer, prompt_path=PROMPTS_PATH, wrap_prompt_as="instruction"
        )
        print(dataset.train)
        print(dataset.train[0])
        PROMPT_TEXT = load_prompting_text(PROMPTS_PATH)

        assert (
            dataset.train[0]["text_q"]
            == "<s>[INST] "
            + PROMPT_TEXT
            + "\nQ: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Answer Choices: (a) ignore\n(b) enforce\n(c) authoritarian\n(d) yell at\n(e) avoid"
            + " [/INST]"
        )

        dataset = DataLoaderFactory.create(
            "commonsense_qa", batch_size=32, tokenizer=tokenizer, prompt_path=PROMPTS_PATH, wrap_prompt_as="context"
        )
        assert (
            dataset.train[0]["text_q"]
            == "<s>[INST] <<SYS>>\n"
            + PROMPT_TEXT
            + "\n<</SYS>>"
            + "\n\nQ: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Answer Choices: (a) ignore\n(b) enforce\n(c) authoritarian\n(d) yell at\n(e) avoid"
            + " [/INST]"
        )
