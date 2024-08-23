import math
import os
import time

import torch

import src.helper_functions as helper
from src.Probing.hidden_state_extractor import hidden_state_extractor


class probing_dataset_creator(hidden_state_extractor):
    def __init__(
        self,
        LM,
        LM_tokenizer,
        LM_name,
        extracted_layers,
        hidden_state_extraction_type="word level",
        hidden_state_extraction_method="last",
    ):
        super().__init__(LM, LM_tokenizer, LM_name, extracted_layers)
        self.probe_subset_num_examples = (
            None  # number of examples in one subset of the probing dataset
        )

        # either "word level" for word-level tasks, or "sentence level" for sentence-level tasks
        self.hidden_state_extraction_type = hidden_state_extraction_type
        self.hidden_state_extraction_method = (
            hidden_state_extraction_method  # either "last" or "average"
        )
        self.probe_input_data = None
        self.probe_output_data = None
        self.probe_input_example_size = None
        self.num_training_examples = None
        self.probe_data_save_path = None

        try:
            LM_num_layers = LM.config.num_hidden_layers
            self.LM_num_layers = LM_num_layers
        except Exception:
            raise Exception(
                'cannot get number of LM layers using this "LM.config.num_hidden_layers"'
            )

        for layer_num in extracted_layers:
            if not (layer_num >= 0 and layer_num <= LM_num_layers):
                raise Exception(
                    f"requested extracted layer: {layer_num} while LM has {LM_num_layers} layers"
                )

    def get_probe_data_save_dir(self):
        return self.probe_data_save_path

    def split_dataset_to_subsets(self, input_data_list, output_data_list, num_subsets):
        """
        Description:
            datasets may not fit GPU RAM, so this functions splits them into smaller subsets so they can fit RAM
                requirements.
            This is different from splitting the dataset into training, validation, and test datasets
            (This functions splits each of these sets into smaller subsets)

        Parameters:
            input_data_list: list, list of sentences of the dataset that should be split into smaller subsets
            output_data_list: list, contains the labels for the corresponding sentence/word in input_data_list
            num_subsets: int, denotes number of subsets that the whole dataset will be split into

        Returns:
            input_data_subset_list: list of sublists, each sublist contains some sentences from the dataset
            output_data_subset_list: list of sublists, each sublist contains the labels corresponding to a
                                sentence/word in input_data_subset_list
        """

        if num_subsets == 1:
            self.probe_subset_num_examples = len(input_data_list)
            return [input_data_list], [output_data_list]
        else:
            # number of examples of the dataset
            dataset_len = len(input_data_list)
            # number of examples in each subset
            subset_len = int(math.ceil(dataset_len / num_subsets))
            self.probe_subset_num_examples = subset_len
            # lists that will hold the subsets
            input_data_subset_list = []
            output_data_subset_list = []
            for i in range(num_subsets):
                start_index = i * subset_len
                # for the last subset, number of examples may be smaller than subset_len
                end_index = min((i + 1) * subset_len, len(input_data_list))
                # add subsets to the lists
                input_data_subset_list.append(input_data_list[start_index:end_index])
                output_data_subset_list.append(output_data_list[start_index:end_index])

        return input_data_subset_list, output_data_subset_list

    def add_batch_to_subset(self, batch):
        """
        Description:
            get the hidden representations of the input batch and add that to the tensor that holds all hidden states
            of the current subset

        Parameters:
            batch: list of str, a list of the input sentences

        Returns:
            None
        """
        extracted_hidden_states = self.extract_hidden_states(
            batch,
            self.hidden_state_extraction_type,
            self.hidden_state_extraction_method,
        )

        # add extracted hidden states to tensor that holds all hidden states for the subset
        start_index = (
            (self.probe_input_data == 0).all(dim=-1).nonzero()[0]
        )  # get first index of row with all zeros
        end_index = start_index + extracted_hidden_states.shape[0]
        self.probe_input_data[start_index:end_index, ...] = extracted_hidden_states

        return

    def initialize_probe_input_dataset(self, input_dataset_subset):
        """
        Initialize the "probe_input_dataset" with zeros. This implies obtaining the number of dimensions of this vector.
        """
        # Get the embedding size of the probed LM
        try:
            LM_embedding_size = self.LM.config.hidden_size
        except Exception:
            try:
                LM_embedding_size = self.LM.config.n_embd
            except Exception as e:
                raise Exception(
                    f"Neither config.hidden_size or config.n_embd work for getting embedding size.\n"
                    f"Returned error is: {e}"
                )

        self.probe_input_example_size = len(self.extracted_layers) * LM_embedding_size

        if "word" in self.hidden_state_extraction_type:
            num_probed_examples = len(
                (" ".join(input_dataset_subset)).split()
            )  # Get total number of words for all examples in the subset
        else:
            num_probed_examples = len(input_dataset_subset)  # Number of examples in the subsets

        # Removed '.to(self.device)' as the model handles distribution
        self.probe_input_data = torch.zeros(
            [num_probed_examples, self.probe_input_example_size], requires_grad=False
        )

    def extract_subset_hidden_states(self, input_dataset_subset, batch_size):
        """
        Description:
            get the hidden representations of the LM for each input sentence for a subset of the dataset

        Parameters:
            input_dataset_subset : list of str, list of sentences for the dataset subset
            dataset_num_words: int, total number of words in the sentences of the whole dataset
            device: device which is used for computation (CPU/GPU/TPU)
            batch_size: int, size of the batches of sentences processed at a time
            hidden_state_token_assignment: str, denotes whether a word should be represented by it last subword
                                            representation or average of all corresponding subword representations
                                        This variable should either equal "last" or "average"
            LM_name: str, name of the LM from which the hidden representations are extracted
            label_shift: int, denotes number of positions by which the POS label is shifted
            shift_direction: str, denotes the direction to which the POS labels should be shifted
                                (either "left" or "right")
            LM_embedding_size: int, embedding size/hidden layer size for the probed LM
            LM_num_layers: int, number of layers, including embedding layer, for the probed LM
            LM_probed_layers: list, contains the LM layer numbers which we want to probe

        Returns:
            probe_input_dataset: torch tensor, tensor holding the word hidden representations for all batches
        """

        # initialize dataloader for use in LM inference
        dataloader = torch.utils.data.DataLoader(
            input_dataset_subset, batch_size=batch_size, shuffle=False
        )

        # initialize probe_input_dataset variable to hold the extracted hidden states
        self.initialize_probe_input_dataset(input_dataset_subset)

        # split the subset into batches and extract hidden states for each batch
        for i, batch in enumerate(dataloader):
            # lapped_time_start = (
            #     time.time()
            # )  # used to measure the time taken for processing a single batch
            # add hidden states of the current batch to the probe dataset
            self.add_batch_to_subset(batch)

            # print(
            #     f"batch iteration number {(i + 1)}, time elapsed for processing {batch_size} sentences: "
            #     f"{time.time() - lapped_time_start:.1f} seconds (current total is {(i + 1) * batch_size} examples)"
            # )

        return

    def create_probe_dataset(
        self,
        input_data_list,
        output_data_list,
        batch_size,
        dataset_name: str,
        dataset_purpose: str,
        num_subsets=1,
    ):
        """
        Description:
            create probing dataset with input being a large tensor with each row containing the hidden representation
            for a single input example, and the output is a list of labels corresponding to each input example

        Parameters:
            input_data_list: list, list of strings for the whole dataset
            output_data_list: list, list of output labels for each input example
            batch_size: int, size of the batches of examples processed at a time
            dataset_name: str, name of the dataset investigated
            dataset_purpose: str, takes either "training", "validation", or "test"
            num_subsets: int, denotes number of subsets that the whole dataset will be split into

        Returns:
            None
        """

        if "train" in dataset_purpose.lower():
            self.num_training_examples = len(input_data_list)

        input_data_subset_list, output_data_subset_list = self.split_dataset_to_subsets(
            input_data_list, output_data_list, num_subsets
        )

        for i, (input_subset, output_subset) in enumerate(
            zip(input_data_subset_list, output_data_subset_list)
        ):
            lapped_time_start = (
                time.time()
            )  # used to measure the time taken for processing a single subset

            # extract hidden states for current input subset and store them in the probe_input_dataset variable
            self.extract_subset_hidden_states(input_subset, batch_size)
            # self.probe_input_dataset now holds probe input examples and output_subset is a list of the
            #   corresponding labels

            self.probe_output_data = output_subset

            print(
                f"subset number {(i + 1)}, time elapsed: "
                f"{time.time() - lapped_time_start:.1f} seconds"
            )

            self.save_probing_dataset(dataset_name, dataset_purpose, i)

        return

    def probe_dataset_save_path(self, dataset_name, dataset_purpose):
        if len(self.extracted_layers) == 1:
            self.probe_data_save_path = os.path.join(
                os.getcwd(),
                "data",
                "probe_datasets",
                dataset_name,
                self.LM_name,
                f"layer_{str(int(self.extracted_layers[0]))}",
            )  # convert extracted layers from scalar tensor to int to str
        else:
            self.probe_data_save_path = os.path.join(
                os.getcwd(), "data", "probe_datasets", dataset_name, self.LM_name
            )

        return os.path.join(self.probe_data_save_path, dataset_purpose)

    def save_probing_dataset(self, dataset_name, dataset_purpose, subset_number):
        """
        Description:
            save probing dataset into a zip file. Input is the extracted hidden states and output is the corresponding
            labels.

        Parameters:
            save_dir: str, path to save directory
            subset_number: int, denotes the subset number of the data

        Returns:
            None
        """

        save_dir = self.probe_dataset_save_path(dataset_name, dataset_purpose)
        os.makedirs(save_dir, exist_ok=True)  # create directory if it doesn't exist

        """ save input and output separately, then convert them to a zip file, then delete the individual files """
        # monitor time required for saving the dataset
        save_time = time.time()
        # save the inputs for the dataset
        input_tensor_file_path = os.path.join(
            save_dir, f"input_tensor_{subset_number}.pt"
        )
        helper.save_torch_tensor(self.probe_input_data, input_tensor_file_path)
        # save output labels for the dataset
        try:
            output_subset_tensor = torch.tensor(
                helper.flatten_nested_list(self.probe_output_data)
            )
        except Exception:  # output is a single list and not a list of sublists
            output_subset_tensor = torch.tensor(self.probe_output_data).float()
        output_tensor_file_path = os.path.join(
            save_dir, f"output_tensor_{subset_number}.pt"
        )
        helper.save_torch_tensor(output_subset_tensor, output_tensor_file_path)

        print(f"save time: {(time.time() - save_time) / 60:.1f} minutes")

        return
