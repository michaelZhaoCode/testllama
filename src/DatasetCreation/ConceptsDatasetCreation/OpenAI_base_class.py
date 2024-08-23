from abc import ABC
from abc import abstractmethod
from abc import abstractproperty

from src.DatasetCreation.ConceptsDatasetCreation.constants import (
    estimated_total_num_examples,
)
from src.DatasetCreation.ConceptsDatasetCreation.OpenAI_pricing import (
    openai_pricing_dict,
)


class openai_base(ABC):
    def __init__(
        self, model_name, investigated_concept, investigator_name, prompt_version
    ):
        self.model = model_name
        self.investigated_concept = investigated_concept
        self.investigator_name = investigator_name
        self.prompt_version = prompt_version

        """ for calculating API usage costs """
        self.generation_usage = 0
        self.estimated_total_num_examples = estimated_total_num_examples
        self.generation_prompt_tokens = 0
        self.generation_completion_tokens = 0
        self.estimated_whole_dataset_generation_cost = 0

    @abstractproperty
    def session_id(self):
        pass

    def create_chat_message(self, openai_obj, messages, model, temperature=None):
        if type(self) is openai_base:
            raise NotImplementedError(
                "This method cannot be called on the base class directly."
            )
        if temperature is None:
            response = openai_obj.chat.completions.create(
                model=model, messages=messages, user=self.session_id
            )
        else:
            response = openai_obj.chat.completions.create(
                model=model,
                messages=messages,
                user=self.session_id,
                temperature=temperature,
            )
        return response.choices[0].message.content, response.usage

    def make_OpenAI_API_call(
        self, openai_obj, prompt, assistant_input="", temperature=None
    ):
        # use assistant input to add already generated examples, so that they exist in GPT4's context
        if type(self) is openai_base:
            raise NotImplementedError(
                "This method cannot be called on the base class directly."
            )
        input_message = [
            {"role": "system", "content": prompt},
            {"role": "assistant", "content": assistant_input},
        ]
        prediction, usage = self.call_openai_API(
            openai_obj, input_message, self.model, temperature
        )

        return prediction, usage

    def call_openai_API(self, openai_obj, input_message, model, temperature=None):
        if type(self) is openai_base:
            raise NotImplementedError(
                "This method cannot be called on the base class directly."
            )
        if temperature is None:
            response = openai_obj.chat.completions.create(
                model=model, messages=input_message, user=self.session_id
            )
        else:
            response = openai_obj.chat.completions.create(
                model=model,
                messages=input_message,
                user=self.session_id,
                temperature=temperature,
            )

        return response.choices[0].message.content, response.usage

    def calculate_API_cost(self, openai_model, prompt_tokens, completion_tokens):
        if type(self) is openai_base:
            raise NotImplementedError(
                "This method cannot be called on the base class directly."
            )

        if openai_model in openai_pricing_dict:
            input_price, output_price = openai_pricing_dict[openai_model]
        else:
            print(f"ERROR: MODEL UNKNOWN ({openai_model})")

        return (input_price * prompt_tokens) + (output_price * completion_tokens)

    def calculate_estimated_total_API_cost(
        self, openai_model, prompt_tokens, completion_tokens
    ):
        pass

    @abstractmethod
    def load_prompt(self):
        pass

    @abstractmethod
    def predict(self):
        pass
