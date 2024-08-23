import numpy as np

from src import helper_functions as helper
from src.DatasetCreation.dataset_creator import dataset_creator


class POS_dataset_creator(dataset_creator):
    def __init__(self, input_data_path, output_data_path):
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.input_data, self.output_labels = self.load_POS_dataset_csv()
        self.POS_conversion_dict = self.create_output_label_conversion_dictionary()
        # list that holds the POS tags after being converted to their numerical label
        self.POS_class_label_list = self.convert_POS_tag_to_class_label()

    def load_POS_dataset_csv(self):
        """
        Description:
            load the list of sentences of penn treebank dataset and the corresponding list of sublists of POS tags.
            Files loaded from the "data" folder

        Parameters:
            None

        Returns:
            sent_list: list of str, list of sentences for the penn treebank dataset
            cleaned_POS_list: list of sublists, each sublist contains the POS tags for its corresponding
                                    sentence in sent_list
        """

        # load sentences list
        sent_list = helper.load_csv_to_list(
            self.input_data_path
        )  # returns a list of sublists, each sublist is a single sentence
        sent_list = [sent[0] for sent in sent_list]

        # load POS tags list
        POS_list = helper.load_csv_to_list(self.output_data_path)
        # remove any nan produced from conversion to dataframe
        cleaned_POS_list = []
        for i, sublist in enumerate(POS_list):
            cleaned_POS_list.append(
                [POS_tag for POS_tag in sublist if isinstance(POS_tag, str)]
            )
            # check that length of POS tag sublist is equal to the number of words in its corresponding sentence
            assert len(cleaned_POS_list[i]) == len(
                sent_list[i].split(" ")
            ), f"mismatch between POS_list and sent_list for example {i}"

        return sent_list, cleaned_POS_list

    def create_output_label_conversion_dictionary(self):
        """
        Description:
            function that creates a dictionary that maps each POS tag to a numerical value for use in multiclass
            classification. This function only creates a mapping for POS tags that exist in the used dataset

        Returns:
            POS_dict: dict, dictionary that maps each POS tag to a numerical value for use in multiclass classification
        """

        flattened_POS_list = helper.flatten_nested_list(self.output_labels)
        flattened_POS_arr = np.array(flattened_POS_list)
        # get unique entries in the POS list
        POS_tags = np.unique(flattened_POS_arr)

        POS_dict = {}
        for i, POS_tag in enumerate(POS_tags):
            POS_dict[POS_tag] = i

        return POS_dict

    def get_label_occurrence_prob(self):
        """
        Description:
            computes the occurrence probability of every output label within the dataset

        Parameters:
            # POS_list: list of sublists, each sublist contains the POS tags for a corresponding sentence in the dataset
            None

        Returns:
            label_occurrence_probs: list, list of occurrence probability of every label
        """

        # convert POS_list from list of sublists to a single list
        flattened_POS_list = helper.flatten_nested_list(self.output_labels)
        POS_occurrence_probs = super().get_label_occurrence_prob(flattened_POS_list)

        return POS_occurrence_probs

    def remove_long_sentences(self, sentence_len_cap):
        """
        Description:
            function that removes examples with sentence lengths bigger than a certain threshold

        Parameters:
            sentence_len_cap: int, sentence length threshold beyond which the sentence is removed from the dataset

        Returns:
            X_modified: list, list of input sentences after removing sentences whose lengths are greater than
                            the threshold
            Y_modified: list, output labels for the modified input sentences list
        """

        if sentence_len_cap == 0:
            X_modified = self.input_data
            Y_modified = self.output_labels
        else:
            X_modified = []
            Y_modified = []
            removed_examples = []
            for i in range(len(self.input_data)):
                if len(self.input_data[i].split(" ")) <= sentence_len_cap:
                    X_modified.append(self.input_data[i])
                    Y_modified.append(self.output_labels[i])
                else:
                    removed_examples.append(self.input_data[i])

            print(f"total number of removed examples: {len(removed_examples)}")
            # print("removed examples:\n")
            # for sentence in removed_examples:
            #     print(f'sent_len: {len(sentence.split(" "))}, sent: {sentence}')

            self.input_data = X_modified
            self.output_labels = Y_modified

        return X_modified, Y_modified

    def convert_POS_tag_to_class_label(self, output_labels=None):
        """
        Description:
            function that converts each POS tags to its corresponding numerical value using the mapping dictionary

        Returns:
            POS_class_list: list of lists, list of POS sublists (numerical values)
        """
        if output_labels is None:
            output_labels = self.output_labels
        POS_class_list = []
        for i, POS_sublist in enumerate(output_labels):
            POS_class_sublist = []
            for j, POS_tag in enumerate(POS_sublist):
                POS_class = self.POS_conversion_dict[POS_tag]
                POS_class_sublist.append(POS_class)
            POS_class_list.append(POS_class_sublist)

        assert [
            len(i) == len(j) for i, j in zip(POS_class_list, output_labels)
        ], "error in converting POS tags to numeric labels"

        return POS_class_list

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

        train_dict, valid_dict, test_dict = super().split_dataset_train_valid_test(
            train_split,
            validation_split,
            test_split,
            shuffle=shuffle,
            random_state=random_state,
        )

        train_dict["output"] = self.convert_POS_tag_to_class_label(train_dict["output"])
        valid_dict["output"] = self.convert_POS_tag_to_class_label(valid_dict["output"])
        test_dict["output"] = self.convert_POS_tag_to_class_label(test_dict["output"])

        return train_dict, valid_dict, test_dict
