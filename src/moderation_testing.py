import numpy as np
from prettytable import PrettyTable

from openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
client = OpenAI(organization=OPENAI_ORG_ID, api_key=OPENAI_API_KEY)

LABELS = ["Not flagged", "Flagged for self-harm instruct", "Flagged for sustain instruct"]


class ModerationTester:
    def __init__(self):
        self.confusion_matrix = np.zeros((3, 3), dtype=int)  # 2x2 matrix for binary classification
        self.failed_examples = []

    def call_openai(self, message: str):
        with open("prompt.txt") as file:
            prompt = file.read().format(message)

        input = [
            {"role": "system", "content": prompt},
        ]

        api_params = {
            "model": "gpt-4o",
            "messages": input,
            "temperature": 0
        }
        response = client.chat.completions.create(**api_params)

        return response.choices[0].message.content, response.usage

    def predict(self, examples_list):
        predicted_labels = []
        for string in examples_list:
            print("Running test: ", string)
            moderation, _ = self.call_openai(string)
            print("Response: ", moderation)
            flagged = "flagged" in moderation.lower()

            if flagged:
                if "self harm" in moderation.lower():
                    print("Self harm instruct.")
                    predicted_labels.append(1)
                elif "sustain" in moderation.lower():
                    print("Sustain instruct.")
                    predicted_labels.append(2)

            else:
                print("Not flagged.")
                predicted_labels.append(0)

        return predicted_labels

    def update_confusion_matrix(self, examples_list, actual_labels):
        predicted_labels = self.predict(examples_list)
        for index, (predicted, actual) in enumerate(zip(predicted_labels, actual_labels)):
            self.confusion_matrix[predicted][actual] += 1
            if predicted != actual:
                self.failed_examples.append(
                    (examples_list[index], "Predicted: " + LABELS[predicted], "Actual: " + LABELS[actual]))

    def print_confusion_matrix(self):
        table = PrettyTable()
        fields = ["Moderation\\Actual"] + LABELS
        table.field_names = fields
        for i, row in enumerate(self.confusion_matrix):
            table.add_row([fields[i + 1]] + list(row))
        print(table)
        for example in self.failed_examples:
            print("")
            print(example[0])
            print(example[1], "|", example[2])

    def accuracy(self):
        correct_predictions = np.trace(self.confusion_matrix)
        total_predictions = np.sum(self.confusion_matrix)
        return correct_predictions / total_predictions if total_predictions else 0


nofilter_examples = []
# Open the file in read mode
with open('conversation.txt', 'r') as file:
    # Iterate through each line in the file
    for line in file:
        # Print each line
        nofilter_examples.append(line.strip())
nofilter_actual = [0, ] * len(nofilter_examples)

# self_harm_examples = []
# # Open the file in read mode
# with open('flagselfharm.txt', 'r') as file:
#     # Iterate through each line in the file
#     for line in file:
#         # Print each line
#         self_harm_examples.append(line.strip())
# self_harm_actual = [1, ] * len(self_harm_examples)
#
# sustain_examples = []
# # Open the file in read mode
# with open('flagsustain.txt', 'r') as file:
#     # Iterate through each line in the file
#     for line in file:
#         # Print each line
#         sustain_examples.append(line.strip())
# sustain_actual = [2, ] * len(sustain_examples)

# Example usage
classifier = ModerationTester()
classifier.update_confusion_matrix(nofilter_examples, nofilter_actual)
# classifier.update_confusion_matrix(self_harm_examples, self_harm_actual)
# classifier.update_confusion_matrix(sustain_examples, sustain_actual)
classifier.print_confusion_matrix()
print("Accuracy:", classifier.accuracy())
