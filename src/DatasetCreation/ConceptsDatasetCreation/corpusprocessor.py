import os
import random
import re
import time

from better_profanity import profanity
from openai import OpenAI

from src.DatasetCreation.ConceptsDatasetCreation.constants import (
    classification_model_name,
)
from src.DatasetCreation.ConceptsDatasetCreation.constants import (
    classification_model_temperature,
)
from src.DatasetCreation.ConceptsDatasetCreation.constants import openai_api_key
from src.DatasetCreation.ConceptsDatasetCreation.constants import openai_org_id
from src.DatasetCreation.ConceptsDatasetCreation.OpenAI_base_class import openai_base

# for testing only
TEST_CODE = False

if TEST_CODE:
    GPT_INPUT_NUM_EXAMPLES = 10000
    GPT_OUTPUT_NUM_EXAMPLES = 100
    NUM_USED_CORPUS_LINES = GPT_INPUT_NUM_EXAMPLES * 5
else:
    GPT_INPUT_NUM_EXAMPLES = 1000000
    GPT_OUTPUT_NUM_EXAMPLES = 30000
    NUM_USED_CORPUS_LINES = GPT_INPUT_NUM_EXAMPLES * 10

TOTAL_NUM_CLASSIFIED = 30000
MIN_WORD_LENGTH = 10
MIN_SENTENCE_WORD_LENGTH = 10
MAX_WORD_LENGTH = 50
MAX_COMBINED_LINES = 5
MAX_NUM_QUOTED_EXAMPLES = int(GPT_INPUT_NUM_EXAMPLES / 5)
CHECK_PROFANITY = False
REMOVE_DIALOG = True

random.seed(1)


class CorpusProcessor(openai_base):
    """
    A class for processing textual corpus data.

    Attributes:
        corpus_path (str): The file path to the textual corpus.
        corpus_text (str): The full text of the corpus loaded from the file.
        corpus_sent (list): A list of sentences from the corpus.
    """

    def __init__(
        self,
        corpus_path,
        investigated_concept=None,
        investigator_name=None,
        prompt_version=None,
    ):
        """
        Initializes the CorpusProcessor with the path to a textual corpus.

        Args:
            corpus_path (str): The file path to the corpus.
        """
        super().__init__(
            classification_model_name,
            investigated_concept,
            investigator_name,
            prompt_version,
        )
        self.corpus_path = corpus_path
        self.corpus_text = self.read_corpus()
        self.corpus_sent = self.split_corpus_sentences()[:NUM_USED_CORPUS_LINES]
        self.processed_strings = []
        self.num_quoted_examples = 0

    def session_id(self):
        pass

    def load_prompt(self):
        # prompt = (
        #     'For the following example, state "False" if the text: '
        #     "is an incomplete or incoherent sentence; "
        #     "does not have an actionable verb; "
        #     "appears to contain out of place words or numbers (like chapter titles); "
        #     "contains isbn numbers; "
        #     "focuses on the setting (i.e. does not focus on the character); "
        #     "describes a non-human (i.e. animalistic) character."
        #     'Otherwise,  state "True". \n Example: '
        # )

        # prompt = (
        #     'For the following example, state "True" ONLY IF the text: '
        #     "contains complete and coherent sentences with actionable verbs; "
        #     "does not contain out of place words or numbers (like chapter titles); "
        #     "does not contain isbn numbers; "
        #     "focuses mainly on human subject and not on the surrounding environment/non-human subjects."
        #     'Otherwise,  state "False". \n Example: '
        # )

        # prompt = (
        #     'For the following example, state "True" ONLY IF the text: '
        #     "contains complete and coherent sentences with actionable verbs; "
        #     "does not contain out of place words or numbers (like chapter titles); "
        #     "does not contain isbn numbers; "
        #     "focuses mainly on human actions and interactions, "
        #     "with minimal mention of non-human subjects (such as the surrounding environment, objects, or animals); "
        #     # "Does not contain any part that focuses significantly on non-human subjects."
        #     'Otherwise,  state "False". \n Example: '

        prompt = (
            'Classify the following example as either "True" or "False" based on the given '
            "conditions:"
            "- The text must contain complete and coherent sentences with actionable verbs."
            "- The text must be free of out-of-place words, numbers (like chapter titles), or ISBN numbers."
            "- The text should mainly focus on human subjects and their actions/interactions, not on the surrounding "
            "environment or non-human subjects."
            'classify as "True" only if all of these conditions are met, otherwise, classify as "False".'
            "\nExample: "
        )

        return prompt

    def predict(self):
        pass

    def read_corpus(self) -> str:
        """
        Reads the corpus from the specified file path and stores it as a string.

        Returns:
            str: The content of the corpus file as a single string.
        """
        with open(self.corpus_path) as file:
            content = file.read()
        return content

    def split_corpus_sentences(self) -> list[str]:
        """
        Splits the corpus text into individual sentences based on newline characters.

        Returns:
            list: A list of sentences extracted from the corpus.
        """
        if self.corpus_text:
            sentences = self.corpus_text.split("\n")
            return sentences

    def normalize_corpus_spacing(self):
        """
        Normalizes spacing in corpus sentences and cleans spaces.
        """
        for i in range(len(self.corpus_sent)):
            cleaned_text = re.sub(r'\s+([:?.!,\'"])', r"\1", self.corpus_sent[i])
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            # Use regex to find a word followed by a space and then "n't"
            pattern = r"(\w+)\s+n't"
            # Replace the space with an empty string
            cleaned_text = re.sub(pattern, r"\1n't", cleaned_text)

            self.corpus_sent[i] = cleaned_text

    def count_sentence_lengths(self, input_string):
        # Regular expression to split sentences based on ".", "?", "!"
        # sentence_endings = re.compile(r'[.!?]')
        sentence_endings = re.compile(r"(?:[.!?]|,\'\')")
        sentences = sentence_endings.split(input_string)
        # remove empty elements from sentences list
        sentences = [element for element in sentences if element]
        # remove starting quotations before counting the number of words
        sentences = [s.replace("``", "").strip() for s in sentences]
        # Calculate the length of each sentence and store in the list
        sentence_lengths = [len(sentence.strip().split()) for sentence in sentences]

        return sentence_lengths

    def randomly_combine_sentences(self):
        """
        Randomly combines adjacent sentences in the corpus sentence list to form new sentences.
        corpus-infrastructure
        Ensures an equal number of opening (``) and closing ('') quotation marks.
        Ensures the combined sentence ends properly if it ends with a closing quotation.
        """
        i = 0
        combinations = 0
        while i < len(self.corpus_sent) and combinations < GPT_INPUT_NUM_EXAMPLES:
            throw_string = False
            # Randomly decide the initial number of sentences to combine
            combine_num = random.randint(1, 3)
            combined_string = " ".join(self.corpus_sent[i : i + combine_num])

            # Resolve any opening and closing quotation issues
            first_opening_quote_index = combined_string.find("``")
            first_closing_quote_index = combined_string.find("''")
            # If there is a closing quotation only in the text or if the text has a closing quote before
            #       an opening quote, throw it out
            if (
                first_opening_quote_index == -1 and first_closing_quote_index != -1
            ) or (
                first_closing_quote_index != -1
                and first_opening_quote_index > first_closing_quote_index
            ):
                i += combine_num
                continue

            # Adjust combine_num to ensure equal number of opening and closing quotations
            while combined_string.count("``") > combined_string.count(
                "''"
            ) and i + combine_num < len(self.corpus_sent):
                combine_num += 1
                combined_string = " ".join(self.corpus_sent[i : i + combine_num])
                if combine_num >= MAX_COMBINED_LINES:
                    throw_string = True
                    break

            # Ensure proper punctuation before a closing quotation
            while (
                combined_string.endswith("''")
                and not (combined_string[-3] in {".", "?", "!"})
                and i + combine_num < len(self.corpus_sent)
            ):
                # check that there is no single quote at combined_string[-3]
                if combined_string.endswith("'''") and combined_string[-4] in {
                    ".",
                    "?",
                    "!",
                }:
                    break
                combine_num += 1
                combined_string = " ".join(self.corpus_sent[i : i + combine_num])
                if combine_num >= MAX_COMBINED_LINES:
                    throw_string = True
                    break

            # ignore example if greater or smaller than thresholds or if throw_string is true
            combined_str_sentence_lengths = self.count_sentence_lengths(combined_string)
            if (
                len(combined_string.split()) > MAX_WORD_LENGTH
                or len(combined_string.split()) < MIN_WORD_LENGTH
                or any(
                    element < MIN_SENTENCE_WORD_LENGTH
                    for element in combined_str_sentence_lengths
                )
                or throw_string is True
                or (first_opening_quote_index != -1 and REMOVE_DIALOG is True)
            ):
                pass
            elif (
                first_opening_quote_index != -1
            ):  # add examples with quotes up to quoted examples limit
                if self.num_quoted_examples < MAX_NUM_QUOTED_EXAMPLES:
                    self.num_quoted_examples += 1
                    # Update the corpus list with the newly combined sentence
                    self.processed_strings.append(combined_string)
                    combinations += 1
                else:
                    pass
            else:
                # Update the corpus list with the newly combined sentence
                self.processed_strings.append(combined_string)
                combinations += 1

            i += combine_num

        random.shuffle(self.processed_strings)

        self.save_processed_examples_txt("random_combine_examples")

        return

    def filter_profanity(self):
        """
        Filters out profanity before calling GPT4
        """
        profanity.load_censor_words()
        print("profanity: finished loading censor words")
        examples = self.processed_strings
        filtered_examples = []
        for i, example in enumerate(examples):
            if profanity.contains_profanity(example):
                print(f"profanity filter: example {i}: {example}")
            else:
                filtered_examples.append(example)

            if i % 1000 == 0 and i != 0:
                print(f"profanity: completed {i} examples")

        print(
            f"profanity filter: number of filtered examples: {len(examples) - len(filtered_examples)}"
        )
        self.processed_strings = filtered_examples

        self.save_processed_examples_txt("profanity_processed")
        return

    def filter_nonhuman_incomplete_sentences(self):
        """
        Filters out sentences from the corpus that do not focus on human subjects
        or are incomplete using GPT4.
        """
        openai_obj = OpenAI(organization=openai_org_id, api_key=openai_api_key)
        examples = self.processed_strings
        final_filtered = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        prompt = self.load_prompt()
        for i, example in enumerate(examples):
            if len(final_filtered) >= GPT_OUTPUT_NUM_EXAMPLES:
                break

            prediction, usage = self.make_OpenAI_API_call(
                openai_obj,
                prompt + example,
                temperature=classification_model_temperature,
            )

            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

            if "True" in prediction:
                final_filtered.append(example)

            if i % 100 == 0 and i != 0:
                print(f"GPT4: checked {i+1} examples")
            if i % 1000 == 0 and i != 0:
                total_cost = self.calculate_API_cost(
                    self.model, total_prompt_tokens, total_completion_tokens
                )
                print(f"GPT4: total cost for {i+1} examples: {total_cost:.2f}")
                self.processed_strings = final_filtered
                self.save_processed_examples_txt("gpt_processed")

        total_cost = self.calculate_API_cost(
            self.model, total_prompt_tokens, total_completion_tokens
        )
        estimated_total_cost = self.calculate_estimated_total_API_cost(
            total_cost, GPT_OUTPUT_NUM_EXAMPLES
        )
        print(f"total cost: {total_cost:.2f}")
        print(
            f"estimated cost for classifying {TOTAL_NUM_CLASSIFIED} examples: ${estimated_total_cost:.2f}"
        )
        self.processed_strings = final_filtered
        self.save_processed_examples_txt("gpt_processed")
        return

    def calculate_estimated_total_API_cost(self, total_cost, num_classified):
        return (TOTAL_NUM_CLASSIFIED / num_classified) * total_cost

    def process(self) -> list[str]:
        """
        Processes the corpus by sequentially calling all processing methods

        Returns:
            list: The processed list of sentences.
        """
        start_time = time.time()
        print("removing redundant spaces")
        self.normalize_corpus_spacing()
        print("combining consecutive sentences randomly")
        self.randomly_combine_sentences()
        if CHECK_PROFANITY is True:
            print("filtering profanity")
            profanity_time = time.time()
            self.filter_profanity()
            print(
                f"profanity execution time: {(time.time()-profanity_time)/60:.2f} minutes"
            )
        else:
            print("filtering non-human or incomplete sentences using GPT-4")
            self.filter_nonhuman_incomplete_sentences()
        print(f"text processing time: {(time.time()-start_time)/60:.2f} minutes")

        return self.processed_strings

    def save_processed_examples_txt(self, file_name):
        file_path = os.path.join(
            os.getcwd(),
            "data",
            "concept_probing",
            "generation",
            "example_sentences",
            f"{file_name}.txt",
        )

        with open(file_path, "w") as f:
            for line in self.processed_strings:
                f.write(f"{line}\n")

        return

    def load_processed_examples_txt(self, file_name):
        file_path = os.path.join(
            os.getcwd(),
            "data",
            "concept_probing",
            "generation",
            "example_sentences",
            f"{file_name}.txt",
        )

        # Initialize an empty list to store the lines
        lines = []

        # Open the file in read mode
        with open(file_path) as file:
            # Read each line and strip leading/trailing whitespace
            lines = [line.strip() for line in file.readlines()]

        self.processed_strings = lines

        return

    def printout(self):
        print(self.processed_strings)
        return self.processed_strings


if __name__ == "__main__":
    book_corpus_path = os.path.join(
        os.getcwd(),
        "data",
        "concept_probing",
        "corpora",
        "bookcorpus",
        "books_large_p1.txt",
    )
    corpus = CorpusProcessor(book_corpus_path)
    processed_examples = corpus.process()

    with open(
        "data/concept_probing/generation/example_sentences/examples.txt", "w"
    ) as f:
        for line in processed_examples:
            f.write(f"{line}\n")
