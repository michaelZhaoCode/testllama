import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from prettytable import PrettyTable

from src.DatasetCreation.ConceptsDatasetCreation.constants import (
    classification_model_name,
)
from src.DatasetCreation.ConceptsDatasetCreation.constants import (
    classification_model_temperature,
)
from src.DatasetCreation.ConceptsDatasetCreation.constants import max_API_cost
from src.DatasetCreation.ConceptsDatasetCreation.OpenAI_base_class import openai_base


class single_example_classifier(openai_base):
    def __init__(self, investigated_concept, investigator_name, prompt_version):
        super().__init__(
            classification_model_name,
            investigated_concept,
            investigator_name,
            prompt_version,
        )

        self.generated_dataset_examples_list = None
        self.generated_dataset_labels = None

        self.labelled_dataset_df = None
        self.confusion_matrix = np.zeros((3, 3), dtype=int)
        # for calculating estimated total API usage costs
        self.num_classified_examples = 0

    @property
    def session_id(self):
        return f"LabelClassifier_-{id(self)}"

    def adjust_prompt_investigated_concept(self, prompt):
        return re.sub(r"{investigated_concept}", str(self.investigated_concept), prompt)

    def load_prompt(self, investigated_concept, prompt_version):
        prompt_file_name = f"{investigated_concept}_{prompt_version}.yaml"
        file_path = os.path.join(
            os.getcwd(),
            "src",
            "DatasetCreation",
            "ConceptsDatasetCreation",
            "prompts",
            "dataset_classification",
            self.investigator_name,
            prompt_file_name,
        )
        assert os.path.exists(file_path), (
            f"classification prompt version {prompt_version} for concept {investigated_concept} "
            f"does not exist \n requested file path: {file_path}"
        )
        with open(file_path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
        prompt = data["text"]

        adjusted_prompt = self.adjust_prompt_investigated_concept(prompt)

        return adjusted_prompt

    def predict(self, openai_obj, generated_examples_list, assistant_input=""):
        # use assistant input to add the example to be classified, along with few shot examples if needed
        prompt = self.load_prompt(self.investigated_concept, self.prompt_version)
        predicted_labels = []
        labelled_df = self.labelled_dataset_df
        labels_list = labelled_df[labelled_df.columns[1]].values.tolist()
        # Initialize variables for printing statistics
        for i, example in enumerate(generated_examples_list):
            msgs = [{"role": "system", "name": "classifier", "content": prompt}]
            # prepend few shot examples to the input example
            if len(assistant_input) == 0:
                input_context = example
            else:
                input_context = f"{assistant_input}\n{example}"

            msgs.append(
                {
                    "role": "user",
                    "name": "director",
                    "content": input_context,
                }
            )

            prediction, usage = self.create_chat_message(
                openai_obj=openai_obj,
                messages=msgs,
                model=self.model,
                temperature=classification_model_temperature,
            )
            prediction = self.check_classification(prediction, i)

            # Get confusion matrix data
            self.confusion_matrix[prediction + 1][labels_list[i] + 1] += 1

            predicted_labels.append(prediction)

            # for calculating API call cost
            self.generation_prompt_tokens += usage.prompt_tokens
            self.generation_completion_tokens += usage.completion_tokens

            API_usage_cost_current = self.calculate_API_cost(
                self.model,
                self.generation_prompt_tokens,
                self.generation_completion_tokens,
            )

            # check that the classification API calls didn't exceed the set threshold amount
            if API_usage_cost_current > max_API_cost:
                self.generated_dataset_labels = predicted_labels
                # save labelled dataset
                self.save_prediction_csv(generated_examples_list, predicted_labels)
                raise Exception(
                    f"classification API calls reached maximum amount which is USD {max_API_cost}"
                )

            if i % 100 == 0 and i != 0:
                print(f"number of currently labelled examples: {i + 1}")

        self.print_statistics()

        """ calculate API usage cost """
        API_usage_cost_total = self.calculate_API_cost(
            self.model, self.generation_prompt_tokens, self.generation_completion_tokens
        )
        self.generation_usage = API_usage_cost_total
        self.num_classified_examples = len(generated_examples_list)
        self.estimated_whole_dataset_generation_cost = (
            self.calculate_estimated_total_API_cost(
                self.model,
                self.generation_prompt_tokens,
                self.generation_completion_tokens,
            )
        )

        self.generated_dataset_labels = predicted_labels

        # save labelled dataset
        self.save_prediction_csv(generated_examples_list, predicted_labels)

        return predicted_labels, API_usage_cost_total

    def print_statistics(self):
        # Print a 3x3 confusion matrix
        table = PrettyTable()
        headers = ["Predicted\\Actual", "Negative", "Not Relevant", "Positive"]
        table.field_names = headers
        for i, row in enumerate(self.confusion_matrix):
            table.add_row([f"{headers[i + 1]}"] + [str(x) for x in row])
        print(table)

        # Calculate Precision, Recall, and Accuracy of each class
        # Assuming confusion_matrix is a 2D numpy array
        # Number of classes
        num_classes = self.confusion_matrix.shape[0]

        # Initialize lists to store precision and recall
        precision = []
        recall = []

        # Calculate precision and recall for each class
        for i in range(num_classes):
            true_positive = self.confusion_matrix[i][i]
            false_positive = self.confusion_matrix[:, i].sum() - true_positive
            false_negative = self.confusion_matrix[i, :].sum() - true_positive

            precision_i = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) != 0
                else 0
            )
            recall_i = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) != 0
                else 0
            )

            precision.append(precision_i)
            recall.append(recall_i)

        # Calculate accuracy
        accuracy = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)

        # Print results
        print("Precision for each class:", precision)
        print("Recall for each class:", recall)
        print("Accuracy:", accuracy)

    def check_classification(self, prediction: str, example_num: int):
        # Get the last character of the prediction
        output = prediction.strip()[-1]
        try:
            return int(output)
        except ValueError:
            print(prediction)
            if "-1" in prediction:
                return -1
            elif "0" in prediction:
                return 0
            elif "1" in prediction:
                return 1
            else:
                print(f"Prediction: {prediction}")
                raise Exception(
                    f"Chat outputs classification label that is not in [-1, 0, 1], prediction output: {prediction}"
                )

    def save_prediction_csv(self, generated_examples_list, generated_labels):
        # create dataframe from examples and their generated labels
        generated_df = pd.DataFrame(
            {
                "generated example": generated_examples_list,
                "generated label": generated_labels,
            }
        )

        # set the save directory
        save_dir = self.set_dataset_save_dir(file_extension="csv")
        os.makedirs(save_dir, exist_ok=True)  # create directory if it doesn't exist

        file_name = self.create_dataset_file_name(save_dir)

        file_path = os.path.join(save_dir, f"{file_name}.csv")

        # Save the DataFrame to a CSV file
        generated_df.to_csv(file_path, index=False)

        return

    def set_dataset_save_dir(self, file_extension):
        return os.path.join(
            os.getcwd(),
            "data",
            "concept_probing",
            "classification",
            self.investigated_concept,
            self.investigator_name,
            file_extension,
        )

    def create_dataset_file_name(self, save_dir):
        # Get the current date and time
        date = str(datetime.now().date())

        # investigator_name = self.investigator_name
        prompt_version = f"P{self.prompt_version}"

        # Get the list of all entries in the directory
        entries = os.listdir(save_dir)

        # regex to get the version number without including the extension
        regex_pattern = r"([^_]+)(?=\.[^.]+$)"
        # get the classification version for each classification in the directory
        classification_versions = [
            re.search(regex_pattern, i).group(1) for i in entries
        ]
        classification_versions.sort()
        # get new classification version number
        new_classification_version = (
            0 if not classification_versions else int(classification_versions[-1]) + 1
        )

        file_name = f"{date}_{prompt_version}_{new_classification_version}"

        return file_name

    def load_csv(self, file_path):
        # Initialize an empty list to store the rows
        generated_examples_list = []

        # read csv as pandas dataframe
        generated_examples_df = pd.read_csv(file_path)
        # get the first column which is the input examples
        generated_examples_list = generated_examples_df[
            generated_examples_df.columns[0]
        ].values.tolist()
        self.generated_dataset_examples_list = generated_examples_list
        self.labelled_dataset_df = generated_examples_df
        return generated_examples_list

    def calculate_estimated_total_API_cost(
        self, openai_model, prompt_tokens, completion_tokens
    ):
        estimated_total_prompt_tokens = prompt_tokens * int(
            self.estimated_total_num_examples / self.num_classified_examples
        )
        estimated_total_completion_tokens = completion_tokens * int(
            self.estimated_total_num_examples / self.num_classified_examples
        )

        return self.calculate_API_cost(
            openai_model,
            estimated_total_prompt_tokens,
            estimated_total_completion_tokens,
        )
