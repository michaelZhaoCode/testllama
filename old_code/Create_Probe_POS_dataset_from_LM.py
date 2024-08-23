import gc
import math
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import BertModel
from transformers import BertTokenizerFast
from transformers import GPT2Model
from transformers import GPT2TokenizerFast

# from nltk.corpus import ptb
# from memory_profiler import memory_usage, profile


# def load_POS_dataset(code_test = False):
#     '''
#     Description:
#         This function uses the ptb package from nltk to read the files for the penn treebank dataset (wsj only) and
#         process the dataset so that it consists of 1 sentence per example with each word in the sentence having its
#         corresponding POS tag. These sentences may then be updated so that each word is associated with it hidden
#         representations from a language model.
#         Function saves the list of sentences and list of sublists of POS tags.
#
#     Parameters:
#         code_test : boolean, a flag that toggles between debugging mode and normal running mode. In debugging mode,
#                     only 500 sentences are returned (actual number is slightly above 500)
#
#     Returns:
#         sent_list : list, a list containing the 1 sentence examples from the penn treebank dataset
#         POS_list : list of sublists, each sublist contains the POS tags for its corresponding sentence in sent_list
#     '''
#
#     processing_time_start = time.time()     # used to measure the time taken to process the brown + wsj datasets
#     sent_list = []      # a list of all the sentences (each sentence is a list of words)
#     POS_list = []    # a list of the POS tags for the sentences (there's a list of POS tags for each sentence)
#     for file_id in ptb.fileids():
#         POS_tagged_sents = ptb.tagged_sents(file_id)   # nltk corpus reader containing the sentences with each word
#         and its correspoding POS tag
#         # iterator for the tagged sentences
#         for tagged_sent in POS_tagged_sents:
#             tagged_sent_arr = np.array(tagged_sent)
#             sent = tagged_sent_arr[:, 0].tolist()
#             sent = ' '.join(sent)
#             POS = tagged_sent_arr[:, 1].tolist()
#             sent_list.append(sent)
#             POS_list.append(POS)
#
#         # for debugging purposes of rest of code
#         if code_test == True and len(POS_list) >= 500:
#             print(f"total time elapsed for processing: {time.time() - processing_time_start:.3f} seconds")
#             print(f"size of POS dataset: {len(POS_list)} sentences")
#
#             # save sentence list and POS list
#             save_csv(sent_list, "input_sentences")
#             save_csv(POS_list, "output_POS")
#
#             return sent_list, POS_list
#
#
#     print(f"total time elapsed for processing: {time.time() - processing_time_start:.3f} seconds")
#     print(f"size of POS dataset: {len(POS_list)} sentences")
#
#     # save sentence list and POS list
#     save_csv(sent_list, "input_sentences")
#     save_csv(POS_list, "output_POS")
#
#     return sent_list, POS_list


# @profile
def infer_LM(tokenizer, LM, batch, device):
    """
    Description:
        Run LM on input batch

    Parameters:
        LM: LM used for inference
        tokenizer: tokenizer used to convert sentences into tokens
        batch : list of str, a list of the input sentences
        device: device on which the inference is made (CPU/GPU)

    Returns:
        output: output of the LM when fed with the tokenized input sentences
        tokenized_batch: input batch after tokenization
    """

    # convert sentences into tokens
    tokenized_batch = tokenizer(batch, padding=True, return_tensors="pt").to(device)
    # run LM on tokenized batch
    with torch.no_grad():
        LM.eval()
        output = LM(**tokenized_batch)

    return output, tokenized_batch


def infer_LM_with_att(LM_name, batch, device):
    """
    Description:
        Instantiate LM along with its tokenizer and run it on the input batch to get the attention weights

    Parameters:
        LM_name: str, name of the LM from which the hidden representations are extracted
        batch : list of str, a list of the input sentences
        device: device on which the inference is made (CPU/GPU)

    Returns:
        output: output of the LM when fed with the tokenized input sentences
        tokenized_batch: input batch after tokenization
    """

    if "bert" in LM_name:
        LM = BertModel.from_pretrained(LM_name, output_attentions=True).to(device)
        tokenizer = BertTokenizerFast.from_pretrained(LM_name)
    elif "gpt2" in LM_name:
        LM = GPT2Model.from_pretrained(LM_name, output_attentions=True).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(LM_name)
        tokenizer.pad_token = tokenizer.eos_token
    # convert sentences into tokens
    batch = batch[0]
    tokenized_batch = tokenizer(batch, padding=True, return_tensors="pt").to(device)
    # run LM on tokenized batch
    with torch.no_grad():
        LM.eval()
        output = LM(**tokenized_batch)
        # att_df = torch.squeeze(output["attentions"][-1]).numpy()

    return output, tokenized_batch


# @profile
def create_cls_sep_pad_mask(tokenized_batch):
    """
    Description:
        create a mask to zero out any [cls], [sep], and padding tokens (used for BERT models)

    Parameters:
        tokenized_batch : batch of sentences after tokenization

    Returns:
        combined_mask: torch tensor, mask that zeros out all unnecessary tokens
    """

    # use the padding mask to zero out any padded hidden representations
    padding_mask = tokenized_batch.data["attention_mask"].view(-1)
    # for BERT models, zero out the [cls] and [sep] by adding their positions to the padding mask
    # masks have to have float32 datatype to avoid allocating extra memory when being multiplied by any tensor
    cls_mask = (tokenized_batch.data["input_ids"].view(-1) != 101).float()
    sep_mask = (tokenized_batch.data["input_ids"].view(-1) != 102).float()
    # combine the 3 masks
    combined_mask = torch.mul(torch.mul(cls_mask, sep_mask), padding_mask)

    return combined_mask


def create_pad_mask(tokenized_batch):
    """
    Description:
        create a mask to zero out padding tokens (used for models that don't have "cls" and "sep" tokens)

    Parameters:
        tokenized_batch : batch of sentences after tokenization

    Returns:
        combined_mask: torch tensor, mask that zeros out all unnecessary tokens
    """

    # use the padding mask to zero out any padded hidden representations
    padding_mask = tokenized_batch.data["attention_mask"].view(-1)

    return padding_mask


# @profile
def process_LM_hidden_states(
    LM_output,
    tokenized_batch,
    device,
    LM_name="bert-base-uncased",
    LM_probed_layers=None,
):
    """
    Description:
        remove all [cls], [sep], and padding vectors from hidden states vector

    Parameters:
        LM_output : output of the LM for a batch
        tokenized_batch: batch input sentences after tokenization
        device: device which is used for computation (CPU/GPU/TPU)
        LM_name: str, name of the LM from which the hidden representations are extracted
        LM_probed_layers: list, contains the LM layer numbers which we want to probe

    Returns:
        processed_hidden_states_no_overhead: processed hidden states vector
    """

    # get hidden states from all layers of the LM for every input sentence in the batch
    hidden_states_tuple = (
        LM_output.hidden_states
    )  # tuple of tensors. num_tuples = num_layers, each layer is a tensor with shape (batch_size, seq_len,
    #                                                                                                embedding_size)
    # convert hidden states from tuple of tensors to tensor of tensors (place num_layers dimension beside the
    #           embedding_size dimension so that they can be later joined into one dimension)
    hidden_states = torch.squeeze(
        torch.stack(hidden_states_tuple, dim=-2)
    )  # shape: (batch_size, seq_len, num_layers, embedding_size)
    if LM_probed_layers is None or LM_probed_layers == []:
        pass
    else:
        temp_hidden_states = hidden_states[:, :, LM_probed_layers, ...]
        for i in range(len(LM_probed_layers)):
            assert torch.all(
                torch.eq(
                    temp_hidden_states[:, :, i, ...],
                    hidden_states[:, :, LM_probed_layers[i], ...],
                )
            ), "mismatch between original tensor and its subsest"

        hidden_states = temp_hidden_states  # shape: (batch_size, seq_len, len(LM_probed_layers), embedding_size)

        # delete all unnecessary tensors
        del temp_hidden_states

    # stack the hidden states for the words of all sentences, and stack the number of layers with the embedding size.
    # this means that there is a 1-D vector for every subword
    hidden_states = hidden_states.view(
        hidden_states.shape[0] * hidden_states.shape[1], -1
    )  # shape: (batch_size*seq_len, num_layers*embedding_size)
    if "bert" in LM_name:
        # create a mask to remove [cls], [sep], and padding tokens
        combined_mask = create_cls_sep_pad_mask(
            tokenized_batch
        )  # shape: (batch_size*seq_len)
    elif "gpt2" in LM_name:
        # create a mask to remove padding tokens
        combined_mask = create_pad_mask(tokenized_batch)  # shape: (batch_size*seq_len)

    # repeat the mask so that it has the same shape as that of the hidden states
    # "repeat" allocates double the memory required for the tensor for some reason, "expand" doesn't allocate any new
    #                       memory at all (just gets references to the original tensor)
    # adjusted_mask has to be of the same datatype of hidden_states so as not to allocate more memory than
    #                       necessary in the elementwise multiplication
    adjusted_mask = torch.unsqueeze(combined_mask, -1).expand(
        hidden_states.shape[0], hidden_states.shape[-1]
    )  # shape: (batch_size*seq_len, num_layers*embedding_size)
    # apply the mask to the hidden representation tensor
    processed_hidden_states = torch.mul(
        hidden_states, adjusted_mask
    )  # shape: (batch_size*seq_len, num_layers*embedding_size)
    # remove rows with all zero columns ([cls], [sep], and padded hidden representations)
    processed_hidden_states_no_overhead = processed_hidden_states[
        torch.any(processed_hidden_states != 0, dim=-1)
    ]

    # delete all unnecessary tensors
    del hidden_states_tuple
    del hidden_states
    del combined_mask
    del adjusted_mask
    del processed_hidden_states
    gc.collect()
    torch.cuda.empty_cache()

    return processed_hidden_states_no_overhead


def get_last_subword_hidden_state(
    batch,
    batch_num,
    tokenized_batch,
    processed_hidden_states,
    tokenizer,
    device,
    LM_name="bert-base-uncased",
    label_shift=0,
    shift_direction="left",
):
    """
    Description:
        every word in the sentence is split into a subword, each having its own token and hidden state vector.
        For probing, there needs to be only 1 hidden state per output label, so we select the hidden state of the last
        subword of a word as the hidden state for which there is an output label

    Parameters:
        batch : list of str, a list of the input sentences
        batch_num: int, batch number. Used for verification purposes
        tokenized_batch: batch input sentences after tokenization
        processed_hidden_states: hidden states vector after removing [cls], [sep], and padding vectors
        tokenizer: tokenizer used to convert sentences into tokens
        device: device which is used for computation (CPU/GPU/TPU)
        LM_name: str, name of the LM from which the hidden representations are extracted
        label_shift: int, denotes number of positions by which the POS label is shifted. Used for discarding the last
                    words in each sentence for a correct shift in labels.
        shift_direction: str, denotes the direction to which the POS labels should be shifted (either "left" or "right")

    Returns:
        last_subword_hidden_states: hidden states for the last subword of each word in the sentences of the batch
    """

    # only for verification, get a list of the subword tokens for all words in all sentences in the batch
    if "bert" in LM_name:
        # create a mask to remove [cls], [sep], and padding tokens
        combined_mask = create_cls_sep_pad_mask(
            tokenized_batch
        )  # shape: (batch_size*seq_len)
    elif "gpt2" in LM_name:
        # create a mask to remove padding tokens
        combined_mask = create_pad_mask(tokenized_batch)  # shape: (batch_size*seq_len)

    # this is stacked into a 1-D vector, and the [cls], [sep], and padding tokens are removed from it ()
    processed_subword_tokens = torch.mul(
        tokenized_batch.data["input_ids"].view(-1), combined_mask
    )  # shape: (batch_size*seq_len)
    # remove zeros from tensor
    processed_subword_tokens = processed_subword_tokens[processed_subword_tokens != 0]

    token_counter = 0
    last_subword_indices = (
        []
    )  # list that stores index of the last subword of the words in every sentence
    for i, input_example in enumerate(batch):
        input_example = input_example.split(" ")
        # iterate over every word in the sentence and exclude the last u words corresponding to label_shift = u
        # we still have to iterate over every word in the sentence to properly increment first_subword_index and
        #                   last_subword_index
        for j in range(len(input_example)):
            # in case the word isn't placed at the beginning of the sentence, it is prepended with a space. This
            # gives it a different token ID when tokenized with a GPT2 tokenizer than when tokenized without a space
            # (BERT tokenizers neglect this space)
            if j == 0:
                word = input_example[j]
            else:
                word = " " + input_example[j]

            if "bert" in LM_name:
                # get subword tokenization while removing [cls] and [sep] token
                tokenized_word = tokenizer(word).data["input_ids"][1:-1]
            elif "gpt2" in LM_name:
                # get subword tokenization
                tokenized_word = tokenizer(word).data["input_ids"]

            first_subword_index = (
                token_counter  # first subword index in processed_hidden_states
            )
            last_subword_index = (
                token_counter + len(tokenized_word) - 1
            )  # last subword index in processed_hidden_states
            # check that first and last indices computed actually correspond to first and last indices
            # in the hidden states vector by checking that the subword tokens are the same
            tokenized_word = torch.tensor(tokenized_word).to(device)
            assert torch.all(
                torch.eq(
                    processed_subword_tokens[
                        first_subword_index : last_subword_index + 1
                    ],
                    tokenized_word,
                )
            ), "extracted indices don't match the indices in the adjusted subword tokens vector"
            # only add the index if it falls within the shifted window
            if (
                shift_direction == "left" and j < (len(input_example) - label_shift)
            ) or (shift_direction == "right" and j >= label_shift):
                last_subword_indices.append(last_subword_index)
            token_counter = last_subword_index + 1

            # calling the garbage collector too frequently consumes a lot of time
            # del tokenized_word
            # gc.collect()
            # torch.cuda.empty_cache()

    # extract the rows that have the last subword hidden states
    last_subword_hidden_states = torch.squeeze(
        processed_hidden_states[[last_subword_indices], ...]
    )

    # # check that the extraction is done correctly (verification is done only on the first batch)
    # if batch_num == 0:
    #     for i, index in enumerate(last_subword_indices):
    #         assert torch.equal(last_subword_hidden_states[i, ...], processed_hidden_states[index, ...]),
    #               "mismatch between extracted rows"

    # check that the extraction is done correctly
    for i, index in enumerate(last_subword_indices):
        assert torch.equal(
            last_subword_hidden_states[i, ...], processed_hidden_states[index, ...]
        ), "mismatch between extracted rows"

    del tokenized_word
    del combined_mask
    del processed_subword_tokens
    gc.collect()
    torch.cuda.empty_cache()

    return last_subword_hidden_states


# @profile
def extract_batch_hidden_states(
    tokenizer, LM, batch, device, LM_name="bert-base-uncased", LM_probed_layers=None
):
    """
    Description:
        get the hidden representations of the LM for each input sentence for a single batch

    Parameters:
        tokenizer : tokenizer used to convert sentences into tokens
        LM: LM used for inference
        batch: list of str, a list of the input sentences
        device: device which is used for computation (CPU/GPU/TPU)
        LM_name: str, name of the LM from which the hidden representations are extracted
        LM_probed_layers: list, contains the LM layer numbers which we want to probe

    Returns:
        processed_hidden_states: processed hidden states tensor
        tokenized_batch: input batch after tokenization
    """

    # infer_LM_with_att(LM_name, batch, device)
    # run the LM in inference mode
    LM_output, tokenized_batch = infer_LM(tokenizer, LM, batch, device)

    # get the hidden states vector for each batch example and remove from them all [cls], [sep], and padding vectors
    processed_hidden_states = process_LM_hidden_states(
        LM_output,
        tokenized_batch,
        device,
        LM_name=LM_name,
        LM_probed_layers=LM_probed_layers,
    )

    # delete all unnecessary tensors
    del LM_output
    gc.collect()
    torch.cuda.empty_cache()

    return processed_hidden_states, tokenized_batch


# @profile
def extract_word_hidden_states(
    hidden_state_token_assignment,
    batch,
    batch_iteration_num,
    tokenized_batch,
    processed_hidden_states,
    tokenizer,
    device,
    LM_name="bert-base-uncased",
    label_shift=0,
    shift_direction="left",
):
    """
    Description:
        get the hidden representation of a word from its subword token representations. Done either by selecting
        last subword hidden representation or by averaging the hidden representations for all subwords

    Parameters:
        hidden_state_token_assignment : str, denotes whether a word should be represented by it last subword
                                        representation or average of all corresponding subword representations
        batch: list of str, a list of the input sentences
        batch_iteration_num: int, iteration number of current batch
        tokenized_batch: input batch after tokenization
        processed_hidden_states: torch tensor, hidden states tensor for whole batch after removing [cls], [sep]
                                representations and all padded representations
        tokenizer : tokenizer used to convert sentences into tokens
        device: device which is used for computation (CPU/GPU/TPU)
        LM_name: str, name of the LM from which the hidden representations are extracted
        label_shift: int, denotes number of positions by which the POS label is shifted
        shift_direction: str, denotes the direction to which the POS labels should be shifted (either "left" or "right")

    Returns:
        word_hidden_states: torch tensor, hidden states for each word in the sentences of the batch
    """

    if hidden_state_token_assignment == "last":
        # for each word, assign the hidden state of the last subword as the hidden state for the whole word
        word_hidden_states = get_last_subword_hidden_state(
            batch,
            batch_iteration_num,
            tokenized_batch,
            processed_hidden_states,
            tokenizer,
            device,
            LM_name=LM_name,
            label_shift=label_shift,
            shift_direction=shift_direction,
        )
    else:
        # for each word, assign the average of the hidden states of the subword as the hidden state for the whole word
        assert False, "hidden states averaging not implemented yet"

    return word_hidden_states


# @profile
def add_batch_to_probe_dataset(
    tokenizer,
    LM,
    batch,
    batch_iteration_num,
    device,
    hidden_state_token_assignment,
    probe_input_dataset,
    label_shift=0,
    shift_direction="left",
    LM_name="bert-base-uncased",
    LM_probed_layers=None,
):
    """
    Description:
        get the hidden representation of a word from its subword token representations. Done either by selecting
        last subword hidden representation or by averaging the hidden representations for all subwords

    Parameters:
        tokenizer : tokenizer used to convert sentences into tokens
        LM: LM used for inference
        batch: list of str, a list of the input sentences
        batch_iteration_num: int, iteration number of current batch
        device: device which is used for computation (CPU/GPU/TPU)
        hidden_state_token_assignment : str, denotes whether a word should be represented by it last subword
                                        representation or average of all corresponding subword representations
        probe_input_dataset: torch tensor, tensor holding the word hidden representations for all batches
        label_shift: int, denotes number of positions by which the POS label is shifted
        shift_direction: str, denotes the direction to which the POS labels should be shifted (either "left" or "right")
        LM_name: str, name of the LM from which the hidden representations are extracted
        LM_probed_layers: list, contains the LM layer numbers which we want to probe

    Returns:
        probe_input_dataset: torch tensor, tensor holding the word hidden representations for all batches
    """

    # get hidden states for the subwords of all words in each sentence in the batch
    processed_hidden_states, tokenized_batch = extract_batch_hidden_states(
        tokenizer, LM, batch, device, LM_name=LM_name, LM_probed_layers=LM_probed_layers
    )

    # get hidden representations of all words in each sentence in the batch
    word_hidden_states = extract_word_hidden_states(
        hidden_state_token_assignment,
        batch,
        batch_iteration_num,
        tokenized_batch,
        processed_hidden_states,
        tokenizer,
        device,
        LM_name=LM_name,
        label_shift=label_shift,
        shift_direction=shift_direction,
    )

    # get first index of row with all zeros
    start_index = (probe_input_dataset == 0).all(dim=-1).nonzero()[0]
    end_index = start_index + word_hidden_states.shape[0]
    probe_input_dataset[start_index:end_index, ...] = word_hidden_states

    # delete any unnecessary tensor
    del processed_hidden_states
    del tokenized_batch
    del word_hidden_states
    gc.collect()
    torch.cuda.empty_cache()

    return probe_input_dataset


# @profile
def extract_hidden_states(
    dataset,
    dataset_num_words,
    device,
    batch_size=32,
    hidden_state_token_assignment="last",
    LM_name="bert-base-uncased",
    label_shift=0,
    shift_direction="left",
    LM_embedding_size=768,
    LM_num_layers=13,
    LM_probed_layers=None,
):
    """
    Description:
        get the hidden representations of the LM for each input sentence for the whole datasest

    Parameters:
        dataset : list of str, list of sentences for the whole dataset
        dataset_num_words: int, total number of words in the sentences of the whole dataset
        device: device which is used for computation (CPU/GPU/TPU)
        batch_size: int, size of the batches of sentences processed at a time
        hidden_state_token_assignment: str, denotes whether a word should be represented by it last subword
                                        representation or average of all corresponding subword representations
                                    This variable should either equal "last" or "average"
        LM_name: str, name of the LM from which the hidden representations are extracted
        label_shift: int, denotes number of positions by which the POS label is shifted
        shift_direction: str, denotes the direction to which the POS labels should be shifted (either "left" or "right")
        LM_embedding_size: int, embedding size/hidden layer size for the probed LM
        LM_num_layers: int, number of layers, including embedding layer, for the probed LM
        LM_probed_layers: list, contains the LM layer numbers which we want to probe

    Returns:
        probe_input_dataset: torch tensor, tensor holding the word hidden representations for all batches
    """

    assert (
        hidden_state_token_assignment == "last"
        or hidden_state_token_assignment == "average"
    ), "the variable hidden_state_token_assignment is taking an unknown string value"
    # initialize the LM along with its tokenizer
    # to get number of layers: LM.config.num_hidden_layers      (this doesn't include the embedding layer)
    # to get size of embedding vector: LM.config.hidden_size
    if "bert" in LM_name:
        LM = BertModel.from_pretrained(LM_name, output_hidden_states=True).to(device)
        tokenizer = BertTokenizerFast.from_pretrained(LM_name)
    elif "gpt2" in LM_name:
        LM = GPT2Model.from_pretrained(LM_name, output_hidden_states=True).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(LM_name)
        tokenizer.pad_token = tokenizer.eos_token

    # initialize dataloader for use in LM inference
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    # instantiate the tensor that is going to hold the LM hidden states
    if LM_probed_layers is None or LM_probed_layers == []:
        # use entire column of hidden states of the LM
        probe_input_example_size = LM_num_layers * LM_embedding_size
    else:
        # use hidden states from chosen layers in the LM
        probe_input_example_size = len(LM_probed_layers) * LM_embedding_size
    probe_input_dataset = torch.zeros(
        [dataset_num_words, probe_input_example_size], requires_grad=False
    ).to(device)
    # probe_input_dataset = torch.empty([0, probe_input_example_size], requires_grad=False).to(device)

    for i, batch in enumerate(dataloader):
        lapped_time_start = (
            time.time()
        )  # used to measure the time taken for processing a single batch
        # add word hidden representations of the current batch to the probe dataset
        probe_input_dataset = add_batch_to_probe_dataset(
            tokenizer,
            LM,
            batch,
            i,
            device,
            hidden_state_token_assignment,
            probe_input_dataset,
            label_shift=label_shift,
            shift_direction=shift_direction,
            LM_probed_layers=LM_probed_layers,
            LM_name=LM_name,
        )

        print(
            f"batch iteration number {(i+1)}, time elapsed for processing {batch_size} sentences: "
            f"{time.time() - lapped_time_start} seconds (current total is {(i+1)*batch_size})"
        )

    # delete any unnecessary variables
    del LM
    del tokenizer
    del dataloader
    gc.collect()
    torch.cuda.empty_cache()

    return probe_input_dataset


def create_Probe_dataset(
    sent_list,
    POS_list,
    POS_conversion_dict,
    batch_size,
    device,
    set_name,
    LM_name="bert-base-uncased",
    subset_num=0,
    label_shift=0,
    shift_direction="left",
    LM_embedding_size=768,
    LM_num_layers=13,
    LM_probed_layers=None,
    use_control_dataset=False,
    POS_occurrence_probs=None,
    randomize_labels=False,
):
    """
    Description:
        create probing dataset with input being a large tensor with each row containing the hidden representations of a
        word, and the output is a list for which each row is the POS tag corresponding to the word in that same row for
        the input tensor

    Parameters:
        sent_list : list of str, list of sentences for the whole dataset
        POS_list: list of sublists, each sublist contains the POS tags for its corresponding sentence in sent_list
        POS_conversion_dict: dict, dictionary that maps each POS tag to a numerical value for use in multiclass
                                classification
        batch_size: int, size of the batches of sentences processed at a time
        device: device which is used for computation (CPU/GPU/TPU)
        set_name: str, name of the dataset being processed (either "training", "validation", or "test")
        LM_name: str, name of the LM from which the hidden representations are extracted
        subset_num: int, number of the subset in case the main set is split into smaller subsets
                ("main set" here refers to either training, validation, or test sets, not the absolute whole dataset)
        label_shift: int, denotes number of positions by which the POS label is shifted
        shift_direction: str, denotes the direction to which the POS labels should be shifted
                        (either "left" or "right")
        LM_embedding_size: int, embedding size/hidden layer size for the probed LM
        LM_num_layers: int, number of layers, including embedding layer, for the probed LM
        LM_probed_layers: list, contains the LM layer numbers which we want to probe
        use_control_dataset: bool, controls the creation of the control dataset, which is a non-shifted dataset that
                             has the same number of training/validtion/test examples as that of the shifted dataset
        POS_occurrence_probs: dict, dictionary containing every POS tag along with its occurrence probability
        randomize_labels: bool, flag which controls whether to randomize the labels for the examples in
                            the probe dataset

    Returns:
        None
    """
    assert (
        shift_direction == "left" or shift_direction == "right"
    ), 'shift direction is assigned with an unkown string. It can only be assigned with "left" or "right"'
    # get number of sentences in dataset
    # num_sentences = len(sent_list)

    # apply shifting to POS_list (does no change when label_shift = 0)
    updated_POS_list = shift_POS_labels(
        POS_list, label_shift=label_shift, shift_direction=shift_direction
    )
    # create a non-shifted output dataset with first/last "label_shift" positions removed. This acts as a control
    # dataset that has exactly the same number of training/test examples as the shifted dataset
    # control dataset must be shifted in the opposite direction of the shifted dataset
    control_shift_direction = "right" if shift_direction == "left" else "left"
    control_POS_list = shift_POS_labels(
        POS_list, label_shift=label_shift, shift_direction=control_shift_direction
    )

    # convert POS_list from list of sublists to a single list
    flattened_POS_list = flatten_nested_list(updated_POS_list)
    # convert control_POS_list from list of sublists to a single list
    flattened_control_POS_list = flatten_nested_list(control_POS_list)

    assert len(flattened_POS_list) == len(
        flattened_control_POS_list
    ), "number of examples for shifted dataset and control dataset are not equal"

    # get number of words for the dataset (corresponds to number of examples in the dataset)
    # in case label_shift = u, we discard the hidden representations for the last u words in each sentence
    dataset_num_words = len(flattened_POS_list)
    # dataset_num_words = len(flattened_POS_list) - label_shift*num_sentences
    # for monitoring the time needed for processing the dataset
    extraction_start_time = time.time()
    # create the probing dataset
    probe_input_data = extract_hidden_states(
        sent_list,
        dataset_num_words,
        device,
        batch_size,
        LM_name=LM_name,
        label_shift=label_shift,
        shift_direction=shift_direction,
        LM_embedding_size=LM_embedding_size,
        LM_num_layers=LM_num_layers,
        LM_probed_layers=LM_probed_layers,
    )
    print(
        f"total time for extracting hidden states: {(time.time() - extraction_start_time) / 60} minutes"
    )

    if label_shift == 0:
        input_data_file_name = set_name + "_probe_input_" + str(subset_num)
        output_data_file_name = set_name + "_probe_output_" + str(subset_num)
    else:
        input_data_file_name = (
            set_name
            + f"_shift_{shift_direction}_{label_shift}_probe_input_"
            + str(subset_num)
        )
        output_data_file_name = (
            set_name
            + f"_shift_{shift_direction}_{label_shift}_probe_output_"
            + str(subset_num)
        )

    # monitor time required for saving the dataset
    save_time = time.time()
    # save the inputs for the dataset
    save_torch_tensor(probe_input_data, input_data_file_name)

    # # create new dictionary in which the non-existent POS tags (due to shifting) aren't used anymore
    # POS_conversion_dict = create_dataset_dependent_POS_conversion_dictionary(updated_POS_list)
    # convert POS tags to class values
    if randomize_labels is False:
        POS_class_list = convert_POS_tag_to_class(
            flattened_POS_list, POS_conversion_dict
        )
    else:
        POS_class_list = []
        for i in range(len(flattened_POS_list)):
            if POS_occurrence_probs is None:
                random_choice = np.random.choice(len(POS_conversion_dict))
            else:
                random_choice = np.random.choice(
                    len(POS_conversion_dict), p=list(POS_occurrence_probs.values())
                )
            POS_class_list.append(random_choice)

    probe_output_data = torch.tensor(POS_class_list).to(device)
    # save the outputs for the dataset
    save_torch_tensor(probe_output_data, output_data_file_name)

    if use_control_dataset is True:
        # # create new dictionary in which the non-existent POS tags (due to shifting) aren't used anymore
        # control_POS_conversion_dict = create_dataset_dependent_POS_conversion_dictionary(control_POS_list)
        # (for control dataset) convert POS tags to class values
        control_output_data_file_name = (
            set_name + f"_shift_{shift_direction}_0_probe_output_" + str(subset_num)
        )  # file name for non-shifted dataset adjusted to have same number of examples as shifted dataset
        control_POS_class_list = convert_POS_tag_to_class(
            flattened_control_POS_list, POS_conversion_dict
        )
        probe_control_output_data = torch.tensor(control_POS_class_list).to(device)
        # (for control dataset) save the outputs for the dataset
        save_torch_tensor(probe_control_output_data, control_output_data_file_name)
        del probe_control_output_data
    # else:
    #     control_POS_conversion_dict = {}

    print(f"save time: {(time.time() - save_time) / 60} minutes")

    del probe_input_data
    del probe_output_data
    gc.collect()
    torch.cuda.empty_cache()

    return


def load_POS_dataset_csv():
    """
    Description:
        load the list of sentences of penn treebank dataset and the corresponding list of sublists of POS tags.
        Files loaded from the "data" folder

    Parameters:
        None

    Returns:
        sent_list: list of str, list of sentences for the penn treebank dataset
        cleaned_POS_list: list of sublists, each sublist contains the POS tags for its corresponding sentence
                            in sent_list
    """

    # load sentences list
    sent_df_path = os.path.join(
        os.getcwd(),
        "data",
        "POS_probing",
        "input_dataset",
        "input_sentences.csv",
    )
    sent_df = pd.read_csv(sent_df_path, header=None)
    sent_list = sent_df[0].values.tolist()

    # load POS tags list
    POS_df_path = os.path.join(
        os.getcwd(), "data", "POS_probing", "input_dataset", "output_POS.csv"
    )
    POS_df = pd.read_csv(POS_df_path, header=None)
    POS_list = POS_df.values.tolist()
    # remove any nan produced from conversion to dataframe
    cleaned_POS_list = []
    for i, sublist in enumerate(POS_list):
        cleaned_POS_list.append(
            [POS_tag for POS_tag in sublist if isinstance(POS_tag, str)]
        )
        # check that length of POS tag sublist is equal to the number of words in its corresponding sentence
        assert len(cleaned_POS_list[i]) == len(
            sent_list[i].split(" ")
        ), "mismatch between POS_list and sent_list"

    return sent_list, cleaned_POS_list


def load_csv(file_name, index_col=None):
    """
    Description:
        load any csv file from the "data" folder

    Parameters:
        file_name: name of file that should be loaded
        index_col: column whose elements are used as labels for each row

    Returns:
        df: pandas dataframe, contains the information in the loaded csv file
    """

    # file path
    df_path = os.getcwd() + os.path.sep + "data" + os.path.sep + file_name + ".csv"
    # dataframe for file
    df = pd.read_csv(df_path, index_col=index_col, header=None)

    return df


def save_csv(input_list, csv_name, save_index=False):
    """
    Description:
        save a list as a csv file in the "data" folder

    Parameters:
        input_list: list, data that is desired to be saved
        csv_name: string, desired name of file that will hold the saved data
        save_inex: boolean, flag for writing the row index

    Returns:
        None
    """

    # path for saved file
    df_path = os.getcwd() + os.path.sep + "data" + os.path.sep + csv_name + ".csv"
    # convert list to dataframe
    df = pd.DataFrame(input_list)
    # save dataframe to csv
    df.to_csv(df_path, index=save_index, header=False)

    return


def save_torch_tensor(input_tensor, tensor_name):
    """
    Description:
        save a torch tensor in the

    Parameters:
        input_tensor: torch tensor, tensor that is intended to be saved
        tensor_name: string, intended file name for the saved tensor

    Returns:
        None
    """

    folder_path = os.path.join(os.getcwd(), "old_code", "data")
    os.makedirs(folder_path, exist_ok=True)  # create directory if it doesn't exist
    file_name = f"{tensor_name}.pt"
    file_path = os.path.join(folder_path, file_name)
    # saving the tensor
    torch.save(input_tensor, file_path)

    return


def load_probe_dataset(
    device,
    set_name,
    subset_num=0,
    label_shift=0,
    shift_direction="left",
    control_test=False,
):
    """
    Description:
        load the input and output files for the probing dataset from the "data" folder

    Parameters:
        device: device which is used for computation (CPU/GPU/TPU)
        set_name: str, name of the dataset being processed (either "training", "validation", or "test")
        subset_num: int, number of the subset in case the main set is split into smaller subsets
                    ("main set" here refers to either training, validation, or test sets, not the absolute
                            whole dataset)
        label_shift: int, denotes number of positions by which the POS label is shifted
        shift_direction: str, denotes the direction to which the POS labels should be shifted
                    (either "left" or "right")
        control_test: boolean, chooses whether to load the output for the shifted dataset or for the control dataset

    Returns:
        input_data: torch tensor, tensor holding the hidden representations of words
        output_data: torch tensor, tensor holding the POS tags (mapped to numerical values) of the corresponding words
    """

    assert not (
        label_shift == 0 and control_test is True
    ), "no control test should be done when there's no shift in the dataset"
    # file names for input data and output data
    if label_shift == 0:
        input_data_file_name = set_name + "_probe_input_" + str(subset_num)
        output_data_file_name = set_name + "_probe_output_" + str(subset_num)
    else:
        # same input is used for shifted dataset and control dataset
        input_data_file_name = (
            set_name
            + f"_shift_{shift_direction}_{label_shift}_probe_input_"
            + str(subset_num)
        )
        if control_test is False:
            output_data_file_name = (
                set_name
                + f"_shift_{shift_direction}_{label_shift}_probe_output_"
                + str(subset_num)
            )
        else:
            output_data_file_name = (
                set_name + f"_shift_{shift_direction}_0_probe_output_" + str(subset_num)
            )

    # file paths for input data and output data
    input_data_file_path = (
        os.getcwd()
        + os.path.sep
        + "old_code"
        + os.path.sep
        + "data"
        + os.path.sep
        + input_data_file_name
        + ".pt"
    )
    output_data_file_path = (
        os.getcwd()
        + os.path.sep
        + "old_code"
        + os.path.sep
        + "data"
        + os.path.sep
        + output_data_file_name
        + ".pt"
    )
    # load the input and output data
    input_data = torch.load(input_data_file_path, map_location=torch.device(device))
    output_data = torch.load(output_data_file_path, map_location=torch.device(device))

    return input_data, output_data


def flatten_nested_list(nested_list):
    """
    Description:
        convert list of sublists to a single list

    Parameters:
        nested_list: list of sublists

    Returns:
        flattened_list: list
    """
    flattened_list = [item for sublist in nested_list for item in sublist]

    return flattened_list


def split_dataset_subsets(sent_list, POS_list, num_subsets):
    """
    Description:
        datasets may not fit RAM, so this functions splits them into smaller subsets so they can fit RAM requirements.
        This is different from splitting the dataset into training, validation, and test datasets
        (This functions splits each of these sets into smaller subsets)

    Parameters:
        sent_list: list, list of sentences of the dataset that should be split into smaller subsets
        POS_list: list of sublists, each sublists contains the POS tags for the corresponding sentence in sent_list
        num_subsets: int, denotes the number of subsets to which the main set is split

    Returns:
        list_of_sent_subsets: list of sublists, each sublist contains some sentences from the dataset
        list_of_POS_subsets: list of sublists of subsublists, each subsublist contains the POS tags corresponding to a
                            sentence in list_of_sent_subsets
    """

    # number of examples of the dataset
    dataset_len = len(sent_list)
    # number of examples in each subset
    subset_len = int(math.ceil(dataset_len / num_subsets))
    # lists that will hold the subsets
    list_of_sent_subsets = []
    list_of_POS_subsets = []
    for i in range(num_subsets):
        start_index = i * subset_len
        # for the last subset, number of examples may be smaller than subset_len
        end_index = min((i + 1) * subset_len, len(sent_list))
        # add subsets to the lists
        list_of_sent_subsets.append(sent_list[start_index:end_index])
        list_of_POS_subsets.append(POS_list[start_index:end_index])

    return list_of_sent_subsets, list_of_POS_subsets


def create_dataset_dependent_POS_conversion_dictionary(POS_list):
    """
    Description:
        function that creates a dictionary that maps each POS tag to a numerical value for use in multiclass
        classification. This function doesn't create a mapping for POS tags that don't exist in the used dataset

    Parameters:
        POS_list: list of sublists, each sublist contains the POS tags for a corresponding sentence in the dataset

    Returns:
        dataset_POS_dict: dict, dictionary that maps each POS tag to a numerical value for use in multiclass
                        classification
    """

    flattened_POS_list = flatten_nested_list(POS_list)
    flattened_POS_arr = np.array(flattened_POS_list)
    # get unique entries in the POS list
    POS_tags = np.unique(flattened_POS_arr)

    dataset_POS_dict = {}
    for i, POS_tag in enumerate(POS_tags):
        dataset_POS_dict[POS_tag] = i

    return dataset_POS_dict


# def create_full_POS_conversion_dictionary(POS_list):
#     '''
#     Description:
#         function that creates a dictionary that maps each POS tag to a numerical value for use in
#         multiclass classification. This function maps all POS tags, even ones that may not exist in the dataset
#         in case it has POS shifting.
#
#     Parameters:
#         POS_list: list of sublists, each sublist contains the POS tags for a corresponding sentence in the dataset
#
#     Returns:
#         POS_dict: dict, dictionary that maps each POS tag to a numerical value for use in multiclass classification
#     '''
#
#     # convert POS_list from list of sublists to a single list
#     flattened_POS_list = flatten_nested_list(POS_list)
#     # Use Counter to count occurrences of each element in the list
#     POS_occurrences = Counter(flattened_POS_list)
#
#     sorted_POS_occurrences = POS_occurrences.most_common()
#     sorted_POS_occurrences_dict = dict(sorted_POS_occurrences)
#
#     POS_dict = {}
#     for i, (key, value) in enumerate(sorted_POS_occurrences_dict.items()):
#         POS_dict[key] = i
#
#     return POS_dict


def create_full_POS_conversion_dictionary(POS_list):
    """
    Description:
        function that creates a dictionary that maps each POS tag to a numerical value for use in
        multiclass classification. This function maps all POS tags, even ones that may not exist in the dataset
        in case it has POS shifting.

    Parameters:
        POS_list: list of sublists, each sublist contains the POS tags for a corresponding sentence in the dataset

    Returns:
        POS_dict: dict, dictionary that maps each POS tag to a numerical value for use in multiclass classification
    """

    flattened_POS_list = flatten_nested_list(POS_list)
    flattened_POS_arr = np.array(flattened_POS_list)
    # get unique entries in the POS list
    POS_tags = np.unique(flattened_POS_arr)

    POS_dict = {}
    for i, POS_tag in enumerate(POS_tags):
        POS_dict[POS_tag] = i

    return POS_dict


def get_POS_occurrence_prob(POS_list):
    """
    Description:
        computes the occurrence probability of every POS tag within the dataset

    Parameters:
        POS_list: list of sublists, each sublist contains the POS tags for a corresponding sentence in the dataset

    Returns:
        POS_occurrence_probs: list, list of occurrence probability of every POS tag
    """

    # convert POS_list from list of sublists to a single list
    flattened_POS_list = flatten_nested_list(POS_list)
    # Use Counter to count occurrences of each element in the list
    POS_occurrences = Counter(flattened_POS_list)

    sorted_POS_occurrences = POS_occurrences.most_common()
    sorted_POS_occurrences_dict = dict(sorted_POS_occurrences)

    total_num_examples = len(flattened_POS_list)

    # convert number of occurrences to probabilities
    POS_occurrence_probs = {}
    for key, value in sorted_POS_occurrences_dict.items():
        POS_occurrence_probs[key] = value / total_num_examples

    return POS_occurrence_probs


def count_POS_tag_occurrence(POS_list):
    # convert POS_list from list of sublists to a single list
    flattened_POS_list = flatten_nested_list(POS_list)
    # Use Counter to count occurrences of each element in the list
    POS_occurrences = Counter(flattened_POS_list)

    sorted_POS_occurrences = POS_occurrences.most_common()

    # Print the results
    print(f"total number of tags: {len(flattened_POS_list)}")
    for value, count in sorted_POS_occurrences:
        print(f"{value}: {count} times")


def load_POS_conversion_dictionary():
    """
    Description:
        function that loads dictionary that maps POS tags to numerical values for use in multiclass classification

    Parameters:
        None

    Returns:
        POS_conversion_dict: dict, dictionary that maps each POS tag to a numerical value for use in multiclass
                            classification
    """

    # load dictionary as dataframe
    POS_conversion_df = load_csv("POS_conversion_dictionary")
    # convert dataframe to actual dictionary
    POS_conversion_dict = {}
    for index, row in POS_conversion_df.iterrows():
        POS_conversion_dict[row[0]] = row[1]

    return POS_conversion_dict


def shift_POS_labels(POS_list, label_shift=0, shift_direction="left"):
    """
    Description:
        shift the POS tags n positions in a certain direction (left or right)

    Parameters:
        POS_list: list of sublists, each sublist contains the POS tags for its corresponding sentence in sent_list
        label_shift: int, denotes number of positions by which the POS label is shifted
        shift_direction: str, denotes the direction to which the POS labels should be shifted (either "left" or "right")

    Returns:
        shifted_POS_list: list of sublists, each sublist contains the shifted POS tags for its corresponding sentence
                            in sent_list
    """

    if label_shift == 0:
        return POS_list

    shifted_POS_list = []
    output_label_overlap_count = 0
    for POS_sublist in POS_list:
        if shift_direction == "left":
            shifted_POS_sublist = POS_sublist[label_shift:]
            original_POS_sublist = POS_sublist[
                :-label_shift
            ]  # used for getting number of overlapping labels for the same indices
        else:
            shifted_POS_sublist = POS_sublist[:-label_shift]
            original_POS_sublist = POS_sublist[
                label_shift:
            ]  # used for getting number of overlapping labels for the same indices

        shifted_POS_list.append(shifted_POS_sublist)

        # get number of overlaps between shifted and original list (checks, after shifting, whether a certain index
        #           still has the same POS tag)
        assert len(shifted_POS_sublist) == len(
            original_POS_sublist
        ), "error in extracting shifting sublists"
        sublist_overlap_count = sum(
            1 for a, b in zip(shifted_POS_sublist, original_POS_sublist) if a == b
        )
        output_label_overlap_count += sublist_overlap_count

    total_num_examples = len(flatten_nested_list(shifted_POS_list))
    print(
        f"number of overlaps in output labels for shifted dataset: {output_label_overlap_count} "
        f"within {total_num_examples} examples. Ratio = {output_label_overlap_count/total_num_examples:.3f}"
    )

    return shifted_POS_list


def convert_POS_tag_to_class(flattened_POS_list, POS_conversion_dict):
    """
    Description:
        function that converts each POS tags to its corresponding numerical value using the mapping dictionary

    Parameters:
        flattened_POS_list: list, list of POS tags for every word in all sentences in the dataset
        POS_conversion_dict: dict, dictionary that maps each POS tag to a numerical value for use in
                                multiclass classification

    Returns:
        POS_class_list: list, list of POS tags (numerical values) for every word in all sentences in the dataset
    """

    POS_class_list = []
    for i, POS_tag in enumerate(flattened_POS_list):
        POS_class = POS_conversion_dict[POS_tag]
        POS_class_list.append(POS_class)

    return POS_class_list


def remove_long_sentences(sentence_len_cap, X, Y):
    """
    Description:
        function that removes examples with sentence lengths bigger than a certain threshold

    Parameters:
        sentence_len_cap: int, sentence length threshold beyond which the sentence is removed from the dataset
        X: list, list of input sentences
        Y: list, output labels for the input sentences

    Returns:
        X_modified: list, list of input sentences after removing sentences whose lengths are greater than the threshold
        Y_modified: list, output labels for the modified input sentences list
    """

    if sentence_len_cap == 0:
        X_modified = X
        Y_modified = Y
    else:
        X_modified = []
        Y_modified = []
        removed_examples = []
        for i in range(len(X)):
            if len(X[i].split(" ")) <= sentence_len_cap:
                X_modified.append(X[i])
                Y_modified.append(Y[i])
            else:
                removed_examples.append(X[i])

        print(f"total number of removed examples: {len(removed_examples)}")
        print("removed examples:\n")
        for sentence in removed_examples:
            print(f'sent_len: {len(sentence.split(" "))}, sent: {sentence}')

    return X_modified, Y_modified
