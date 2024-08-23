import argparse
import re
from argparse import RawTextHelpFormatter

from openai import OpenAI

from src.DatasetCreation.ConceptsDatasetCreation.constants import openai_api_key
from src.DatasetCreation.ConceptsDatasetCreation.constants import openai_org_id
from src.DatasetCreation.ConceptsDatasetCreation.dataset_generator import (
    single_label_generator_chat,
)
from src.DatasetCreation.ConceptsDatasetCreation.dataset_generator import (
    whole_dataset_generator_chat,
)

parser = argparse.ArgumentParser(
    description="Creating and preprocessing a dataset for training probes",
    formatter_class=RawTextHelpFormatter,
)

# Positional arguments
parser.add_argument(
    "--investigated_concept",
    type=str,
    help='concept inferred from dataset. For -ve examples, input "not_{concept}"',
)
parser.add_argument(
    "--num_examples", type=int, help="number of examples to be generated"
)
parser.add_argument(
    "--investigator_name",
    type=str,
    help="name of the person running the script for naming the file storing the generated examples",
)
parser.add_argument(
    "--prompt_version", type=int, default=0, help="prompt version used for generation"
)
parser.add_argument(
    "--incremental_num_examples",
    type=int,
    default=5,
    help="incremental number of examples used for generating examples iteratively",
)
parser.add_argument(
    "--example_strings_file",
    type=str,
    default="examples.txt",
    help="file name that has examples that should be used to create examples with similar grammatical structure",
)
parser.add_argument(
    "--whole_dataset",
    action="store_true",
    help="(boolean) use both prompts to generate +ve & -ve examples in one go, and shuffle them",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if (
        args.whole_dataset
        and re.search(r"^not_", args.investigated_concept) is not None
    ):
        investigated_concept = re.sub(r"^not_", "", args.investigated_concept)
    else:
        investigated_concept = args.investigated_concept

    generator_openai_obj = OpenAI(organization=openai_org_id, api_key=openai_api_key)
    if not args.whole_dataset:
        generator = single_label_generator_chat(
            investigated_concept,
            args.num_examples,
            args.incremental_num_examples,
            args.investigator_name,
            args.prompt_version,
            args.example_strings_file,
        )
    else:
        generator = whole_dataset_generator_chat(
            investigated_concept,
            args.num_examples,
            args.incremental_num_examples,
            args.investigator_name,
            args.prompt_version,
            args.example_strings_file,
        )

    prediction, usage_cost = generator.predict(generator_openai_obj)

    print(f"API call cost (USD): {usage_cost:.2f}")
    print(
        f"estimated total API cost (USD): {generator.estimated_whole_dataset_generation_cost:.2f}"
    )
