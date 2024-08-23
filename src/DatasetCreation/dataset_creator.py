from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


class dataset_creator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.input_data, self.output_labels = self.load_dataset_from_csv()

    def load_dataset_from_csv(self):
        dataset = pd.read_csv(self.dataset_path)
        input_examples = dataset.iloc[:, 0].tolist()
        labels = dataset.iloc[:, 1].tolist()

        return input_examples, labels

    def get_label_occurrence_prob(self, output_labels=None):
        """
        Description:
            computes the occurrence probability of every output label within the dataset

        Returns:
            label_occurrence_probs: list, list of occurrence probability of every label
        """

        # if type(self) is dataset_creator:
        #     raise NotImplementedError(
        #         "This method cannot be called on the base class directly."
        #     )

        # Use Counter to count occurrences of each element in the list
        if output_labels is None:
            output_labels = self.output_labels
        label_occurrences = Counter(output_labels)

        sorted_label_occurrences = label_occurrences.most_common()
        # sorted_label_occurrences_dict = dict(sorted_label_occurrences)

        total_num_examples = len(output_labels)

        # convert number of occurrences to probabilities
        label_occurrence_probs = {}
        for key, value in sorted_label_occurrences:
            label_occurrence_probs[key] = {
                "count": value,
                "prob": (value / total_num_examples),
            }

        return label_occurrence_probs

    def split_dataset_train_valid_test(
        self,
        train_split,
        validation_split,
        test_split,
        shuffle=True,
        random_state=1,
    ):
        """
        Description:
            splits the dataset into training, validation, and test sets

        Parameters:
            input_data: list, list contains the input examples for the dataset
            output_data: list, list contains the output labels for the dataset
            train_split: int, training split
            validation_split: int, validation split
            test_split: int, test split
            shuffle: bool, determines whether to shuffle the examples when splitting
            random_state: int, seed for the shuffling

        Returns:
            train_dict: dict, contains input examples and output labels for the training set
            valid_dict: dict, contains input examples and output labels for the validation set
            test_dict: dict, contains input examples and output labels for the test set
        """

        assert (
            train_split + validation_split + test_split == 1
        ), "dataset splits do not add up to 1"

        try:
            # split dataset into training, validation, and test sets
            (
                train_input_list,
                temp_input_list,
                train_output_list,
                temp_output_list,
            ) = train_test_split(
                self.input_data,
                self.output_labels,
                stratify=self.output_labels,
                train_size=train_split,
                shuffle=shuffle,
                random_state=random_state,
            )
            (
                validation_input_list,
                test_input_list,
                validation_output_list,
                test_output_list,
            ) = train_test_split(
                temp_input_list,
                temp_output_list,
                stratify=temp_output_list,
                train_size=validation_split / (validation_split + test_split),
                shuffle=shuffle,
                random_state=random_state,
            )
        except (
            Exception
        ):  # in case output labels is a list of lists ==> cannot be stratified
            # split dataset into training, validation, and test sets
            (
                train_input_list,
                temp_input_list,
                train_output_list,
                temp_output_list,
            ) = train_test_split(
                self.input_data,
                self.output_labels,
                train_size=train_split,
                shuffle=shuffle,
                random_state=random_state,
            )
            (
                validation_input_list,
                test_input_list,
                validation_output_list,
                test_output_list,
            ) = train_test_split(
                temp_input_list,
                temp_output_list,
                train_size=validation_split / (validation_split + test_split),
                shuffle=shuffle,
                random_state=random_state,
            )

        train_dict, valid_dict, test_dict = {}, {}, {}
        train_dict["input"] = train_input_list
        train_dict["output"] = train_output_list
        valid_dict["input"] = validation_input_list
        valid_dict["output"] = validation_output_list
        test_dict["input"] = test_input_list
        test_dict["output"] = test_output_list

        return train_dict, valid_dict, test_dict
