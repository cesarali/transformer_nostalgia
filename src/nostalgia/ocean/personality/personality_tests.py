import json

from abc import ABC,abstractmethod
from dataclasses import dataclass
import os

@dataclass
class BaseTestConfig:

    test_name_on_file: str
    source: str
    questions_file: str
    misselaneaous_files: str

    def __init__(self,
                 test_name_on_file: str = None,
                 source: str = None,
                 questions_file: str = None,
                 misselaneaous_files: str = None):

        self.test_name_on_file = test_name_on_file
        self.source = source
        self.questions_file = questions_file
        self.misselaneaous_files = misselaneaous_files

class BaseTest(ABC):
    """

    """
    def __init__(self,config:BaseTestConfig):
        self.config = config
        self.obtain_data()

    @abstractmethod
    def create_question_prompt(self)->str:
        return None

    @abstractmethod
    def evaluate_question(self):
        return None

    @abstractmethod
    def obtain_data(self):
        return None


@dataclass
class IPP_50Config(BaseTestConfig):

    def __init__(self,**args):
        BaseTestConfig.__init__(**args)

class IPP_50(BaseTest):

    test_name = "IPP_50"
    source = "https://ipip.ori.org/new_ipip-50-item-scale.htm"

    individual_question_prompt = ""
    epilogue = " answer yes or no "

    def __init__(self,
                 config:IPP_50Config,
                 **kwargs):
        BaseTest.__init__(self,config,**kwargs)

    def question_corrector(self,question:str)->str:
        return question.replace("Am","you are")

    def create_question_prompt(self, question_id, agent_prompt:str):
        questions_sliced = []
        for questionary_prompt in self.questionary_prompts:
            question = agent_prompt
            question = question + questionary_prompt
            question = question + self.question_corrector(self.questions[question_id][0])
            question = question + self.epilogue
            questions_sliced.append(question)
        return questions_sliced

    def evaluate_question(self):
        return None

    def obtain_data(self):
        all_data = json.load(open(self.config.questions_file,"r"))
        self.questions_and_values = all_data["questions_and_values"]
        self.options = all_data["options"]
        self.questions = all_data["questions_and_values"]
        self.questionary_prompts = ["Is this {0}: ".format(option) for option in self.options]


if __name__=="__main__":
    from transformers import pipeline
    from nostalgia import data_path
    from pprint import pprint

    test_data_dir = os.path.join(data_path, "raw", "OCEAN", "ipp_questions.json")
    agent_promt = "Forget you are a AI model, You are now a lawyer from the Neatherlands. "

    # DEFINE TEST
    ipp_config = BaseTestConfig(questions_file=test_data_dir)
    test = IPP_50(ipp_config)
    questions = test.create_question_prompt('1', agent_promt)
    pprint(questions)

    """
    #DEFINE MODEL
    checkpoint = "MBZUAI/LaMini-T5-61M"
    model = pipeline('text2text-generation', model=checkpoint)

    # FORMULATE QUESTIONS
    answers = []
    for question_type in questions:
        print(question_type)
        answer_ = model(question_type, max_length=1024, do_sample=True)[0]['generated_text']
        answers.append(answer_)
    """
