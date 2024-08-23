import os

import src.helper_functions as helper
from src.Probing.POSProbing.POS_hidden_state_extractor import POS_hidden_state_extractor
from src.Probing.probing_dataset_creator import probing_dataset_creator


class POS_probing_dataset_creator(POS_hidden_state_extractor, probing_dataset_creator):
    def __init__(
        self,
        LM,
        LM_tokenizer,
        LM_name,
        extracted_layers,
        device,
        num_shift_positions=0,
        hidden_state_extraction_type="word level",
        hidden_state_extraction_method="last",
    ):
        POS_hidden_state_extractor.__init__(
            self,
            LM,
            LM_tokenizer,
            LM_name,
            extracted_layers,
            device,
            num_shift_positions,
        )
        probing_dataset_creator.__init__(
            self,
            LM,
            LM_tokenizer,
            LM_name,
            extracted_layers,
            device,
            hidden_state_extraction_type,
            hidden_state_extraction_method,
        )

        self.num_shift_positions = (
            num_shift_positions  # int, absolute value denotes number of positions by
        )
        #       which the POS label is shifted.
        # Sign denotes direction of shifting (+ve means shift to right, -ve means shift to left)

    def initialize_probe_input_dataset(self, input_dataset_subset):
        """
        Description:
            initialize the "probe_input_dataset" with zeros. This implies obtaining the number of dimensions
            of this vector

        Parameters:
            input_dataset_subset: list of str, list of sentences for the dataset subset

        Returns:
            None
        """

        # shift input examples with the specified number of shift positions
        shifted_input_dataset_subset = []
        if self.num_shift_positions == 0:
            shifted_input_dataset_subset = input_dataset_subset
        else:
            for example in input_dataset_subset:
                example_list = example.split(" ")
                if self.num_shift_positions > 0:
                    shifted_example_list = example_list[: -self.num_shift_positions]
                else:
                    shifted_example_list = example_list[-self.num_shift_positions :]
                shifted_input_dataset_subset.append(" ".join(shifted_example_list))
        ################################################################################

        super().initialize_probe_input_dataset(shifted_input_dataset_subset)

        return

    def shift_POS_labels(self, output_POS_list):
        """
        Description:
            shift the POS tags n positions in a certain direction (left or right)

        Parameters:
            output_POS_list: list of sublists, list of POS sublists corresponding to POS tags for each input example

        Returns:
            shifted_POS_list: list of sublists, each sublist contains the shifted POS tags for its
                                corresponding sentence in sent_list
        """

        # num_shift_positions: int, absolute value denotes number of positions by which the POS label is shifted. Sign
        #   denotes direction of shifting (+ve means shift to right, -ve means shift to left)

        if self.num_shift_positions == 0:
            # return self.output_POS_list
            return output_POS_list
        else:
            shifted_POS_list = []
            output_label_overlap_count = (
                0  # used for tracking number of unchanged labels after shifting
            )
            # for POS_sublist in self.output_POS_list:
            for POS_sublist in output_POS_list:
                if self.num_shift_positions > 0:  # shift to right
                    shifted_POS_sublist = POS_sublist[
                        : -self.num_shift_positions
                    ]  # remove last "self.num_shift_positions" elements
                    original_POS_sublist = POS_sublist[
                        self.num_shift_positions :
                    ]  # used for getting number of overlapping labels for the same indices
                else:  # shift to left
                    shifted_POS_sublist = POS_sublist[
                        -self.num_shift_positions :
                    ]  # remove first "-self.num_shift_positions" elements
                    original_POS_sublist = POS_sublist[
                        : self.num_shift_positions
                    ]  # used for getting number of overlapping labels for the same indices

                shifted_POS_list.append(shifted_POS_sublist)

                # get number of overlaps between shifted and original list (checks, after shifting, whether a
                #           certain index still has the same POS tag)
                assert len(shifted_POS_sublist) == len(
                    original_POS_sublist
                ), "error in extracting shifting sublists"
                sublist_overlap_count = sum(
                    1
                    for a, b in zip(shifted_POS_sublist, original_POS_sublist)
                    if a == b
                )
                output_label_overlap_count += sublist_overlap_count

            total_num_examples = len(helper.flatten_nested_list(shifted_POS_list))
            print(
                f"number of unchanged labels after shifting: {output_label_overlap_count} within {total_num_examples} "
                f"examples. Ratio = {output_label_overlap_count / total_num_examples:.2f}"
            )

            return shifted_POS_list

    def create_probe_dataset(
        self,
        input_data_list,
        output_data_list,
        batch_size,
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
            dataset_purpose: str, takes either "training", "validation", or "test"
            num_subsets: int, denotes number of subsets that the whole dataset will be split into

        Returns:
            None
        """

        # apply shifting to POS_list (does no change when label_shift = 0)
        shifted_POS_list = self.shift_POS_labels(output_data_list)
        super().create_probe_dataset(
            input_data_list,
            shifted_POS_list,
            batch_size,
            "POS",
            dataset_purpose,
            num_subsets,
        )

        return

    def probe_dataset_save_path(self, dataset_name, dataset_purpose):
        # in this case, dataset_name will always be "POS" (set from save_probing_dataset method)
        return os.path.join(
            os.getcwd(),
            "data",
            "probe_datasets",
            dataset_name,
            f"shift{self.num_shift_positions}",
            dataset_purpose,
        )
