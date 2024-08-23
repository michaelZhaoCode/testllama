import argparse
from argparse import RawTextHelpFormatter

from openai import OpenAI

from src.DatasetCreation.ConceptsDatasetCreation.constants import openai_api_key
from src.DatasetCreation.ConceptsDatasetCreation.constants import openai_org_id
from src.DatasetCreation.ConceptsDatasetCreation.dataset_classifier import (
    single_example_classifier,
)

parser = argparse.ArgumentParser(
    description="labelling the created dataset for training probes",
    formatter_class=RawTextHelpFormatter,
)

# Positional arguments
parser.add_argument(
    "--investigated_concept",
    type=str,
    help='concept inferred from dataset. For -ve examples, input "not_{concept}"',
)
parser.add_argument(
    "--investigator_name",
    type=str,
    help="name of the person running the script for naming the file storing the generated examples",
)
parser.add_argument(
    "--prompt_version", type=int, default=0, help="prompt version used for generation"
)

if __name__ == "__main__":
    generated_examples_file_path = "data/concept_probing/generation/ambition_2k_v1.csv"

    args = parser.parse_args()

    classifier_openai_obj = OpenAI(organization=openai_org_id, api_key=openai_api_key)
    classifier = single_example_classifier(
        args.investigated_concept, args.investigator_name, args.prompt_version
    )

    generated_examples_list = classifier.load_csv(generated_examples_file_path)
    prediction, usage_cost = classifier.predict(
        classifier_openai_obj, generated_examples_list
    )

    print(f"API call cost (USD): {usage_cost:.2f}")
    print(
        f"estimated total API cost (USD): {classifier.estimated_whole_dataset_generation_cost:.2f}"
    )
