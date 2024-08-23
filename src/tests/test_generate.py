from openai import OpenAI

from src.DatasetCreation.ConceptsDatasetCreation.constants import openai_api_key
from src.DatasetCreation.ConceptsDatasetCreation.constants import openai_org_id
from src.DatasetCreation.ConceptsDatasetCreation.dataset_generator import (
    whole_dataset_generator_chat,
)

investigated_concept = "ambition"
generated_num_examples = 60
incremental_num_examples = 5
investigator_name = "mohamed"
prompt_version = 6
example_strings_file_name = "examples.txt"


generator_openai_obj = OpenAI(organization=openai_org_id, api_key=openai_api_key)


generator = whole_dataset_generator_chat(
    investigated_concept,
    generated_num_examples,
    incremental_num_examples,
    investigator_name,
    prompt_version,
    example_strings_file_name,
)

prediction, usage_cost = generator.predict(generator_openai_obj)
