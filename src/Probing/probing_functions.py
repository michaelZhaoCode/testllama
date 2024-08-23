import os
import time
import torch

from transformers import AutoTokenizer
from transformers import BertTokenizerFast

from src.custom_LMs.custom_BERT import custom_BertModel
from src.custom_LMs.custom_Llama import custom_LlamaForCausalLM
from src.DatasetCreation.dataset_creator import dataset_creator
from src.DatasetCreation.POSDatasetCreation.POS_dataset_creator import (
    POS_dataset_creator,
)
from src.Probing.POSProbing.POS_probe_trainer import POS_probe_trainer
from src.Probing.POSProbing.POS_probing_dataset_creator import (
    POS_probing_dataset_creator,
)
from src.Probing.probe import linear_probe
from src.Probing.probe_trainer import probe_trainer
from src.Probing.probing_dataset_creator import probing_dataset_creator


def load_LM(LM_name, HF_TOKEN=None, devices=None):
    if devices is None:
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        print("=============")
        print(devices)
        print(len(devices))
        print("=============")

    if "bert" in LM_name.lower():
        LM = custom_BertModel.from_pretrained(
            LM_name,
            output_hidden_states=True,
            device_map=None  # Remove device_map and handle devices manually
        )
        LM_tokenizer = BertTokenizerFast.from_pretrained(LM_name)

    elif "llama" in LM_name.lower():
        model_path = os.path.join(os.getcwd(), "LMs", LM_name)

        # Pass the list of devices to custom_LlamaForCausalLM
        LM = custom_LlamaForCausalLM.from_pretrained(
            model_path,
            token=HF_TOKEN,
            output_hidden_states=True,
            devices=devices  # Explicitly set devices
        )
        LM_tokenizer = AutoTokenizer.from_pretrained(LM_name, token=HF_TOKEN)
        LM_tokenizer.pad_token = LM_tokenizer.eos_token

    return LM, LM_tokenizer


def create_sent_probing_dataset(
    LM_name,
    dataset_name,
    dataset_path,
    train_split,
    validation_split,
    test_split,
    extracted_layers,
    num_train_subsets,
    num_validation_subsets,
    num_test_subsets,
    LM_batch_size,
    HF_TOKEN=None,
    probing_dataset_creator_obj=None,
    devices=None,  # New argument to specify devices for model parallelism
):
    # create sentence dataset
    sent_dataset_creator = dataset_creator(dataset_path)

    (
        train_dict,
        valid_dict,
        test_dict,
    ) = sent_dataset_creator.split_dataset_train_valid_test(
        train_split, validation_split, test_split
    )
    ##################################################################

    if probing_dataset_creator_obj is None:
        # load LM & its tokenizer with devices for model parallelism
        LM, LM_tokenizer = load_LM(LM_name, HF_TOKEN, devices=devices)

        #  create probing dataset
        probing_dataset_creator_obj = probing_dataset_creator(
            LM,
            LM_tokenizer,
            LM_name,
            extracted_layers,
            hidden_state_extraction_type="sentence level",
        )
    else:
        probing_dataset_creator_obj.change_extracted_layers(extracted_layers)

    print("extracting hidden states from LM:")
    start_time = time.time()
    probing_dataset_creator_obj.create_probe_dataset(
        train_dict["input"],
        train_dict["output"],
        LM_batch_size,
        dataset_name,
        "training",
        num_train_subsets,
    )
    probing_dataset_creator_obj.create_probe_dataset(
        valid_dict["input"],
        valid_dict["output"],
        LM_batch_size,
        dataset_name,
        "validation",
        num_validation_subsets,
    )
    probing_dataset_creator_obj.create_probe_dataset(
        test_dict["input"],
        test_dict["output"],
        LM_batch_size,
        dataset_name,
        "test",
        num_test_subsets,
    )
    print(
        f"time to extract hidden states: {(time.time() - start_time) / 60:.1f} minutes"
    )
    ##################################################################

    return probing_dataset_creator_obj



def create_POS_probing_dataset(
    LM_name,
    sentence_path,
    POS_path,
    max_sent_len,
    train_split,
    validation_split,
    test_split,
    extracted_layers,
    device,
    num_shift_positions,
    num_train_subsets,
    num_validation_subsets,
    num_test_subsets,
    LM_batch_size,
):
    # load LM & its tokenizer
    LM, LM_tokenizer = load_LM(LM_name)

    # create POS dataset
    POS_data_creator = POS_dataset_creator(sentence_path, POS_path)
    POS_data_creator.remove_long_sentences(max_sent_len)
    # POS_occurrence_probs = POS_data_creator.get_label_occurrence_prob()
    train_dict, valid_dict, test_dict = POS_data_creator.split_dataset_train_valid_test(
        train_split, validation_split, test_split
    )
    ##################################################################

    #  create POS probing dataset
    start_time = time.time()
    probing_dataset_creator = POS_probing_dataset_creator(
        LM,
        LM_tokenizer,
        LM_name,
        extracted_layers,
        device,
        num_shift_positions=num_shift_positions,
    )
    probing_dataset_creator.create_probe_dataset(
        train_dict["input"],
        train_dict["output"],
        LM_batch_size,
        "training",
        num_subsets=num_train_subsets,
    )
    probing_dataset_creator.create_probe_dataset(
        valid_dict["input"],
        valid_dict["output"],
        LM_batch_size,
        "validation",
        num_subsets=num_validation_subsets,
    )
    probing_dataset_creator.create_probe_dataset(
        test_dict["input"],
        test_dict["output"],
        LM_batch_size,
        "test",
        num_subsets=num_test_subsets,
    )
    print(
        f"time to extract hidden states: {(time.time() - start_time) / 60:.1f} minutes"
    )
    ##################################################################

    num_labels = len(POS_data_creator.POS_conversion_dict)
    probe_input_example_size = probing_dataset_creator.probe_input_example_size

    return probe_input_example_size, num_labels


def train_probe(
    dataset_name,
    probe_input_example_size,
    num_labels,
    device,
    probe_batch_size,
    learning_rate,
    EPOCHS,
    early_stopping,
    num_train_subsets,
    num_validation_subsets,
    num_test_subsets,
    num_shift_positions=0,
    probe_data_save_dir=None,
    feature_truncation_method="average",
):
    # instantiate and train model
    probe = linear_probe(probe_input_example_size, num_labels).to(device)
    if "POS" in dataset_name:
        probe_trainer_obj = POS_probe_trainer(
            probe, device, num_shift_pos=num_shift_positions
        )
    else:
        probe_trainer_obj = probe_trainer(
            probe,
            device,
            dataset_name,
            probe_data_save_dir,
            feature_truncation_method=feature_truncation_method,
        )

    probe_trainer_obj.train(
        probe_batch_size,
        learning_rate,
        EPOCHS,
        early_stopping=early_stopping,
        num_train_subsets=num_train_subsets,
        num_validation_subsets=num_validation_subsets,
    )

    avg_test_accuracy = probe_trainer_obj.test_probe(
        probe_batch_size, num_test_subsets=num_test_subsets
    )

    print(f"probe test accuracy: {avg_test_accuracy:.2f}")

    return probe_trainer_obj, avg_test_accuracy


def test_probe(
    dataset_name,
    probe_input_example_size,
    num_labels,
    device,
    probe_batch_size,
    num_test_subsets,
    num_shift_positions=0,
    learning_rate=None,
    feature_truncation_method="average",
):
    # learning rate only used here for naming of output files and results

    # instantiate and test model
    probe = linear_probe(probe_input_example_size, num_labels).to(device)
    if "POS" in dataset_name:
        probe_trainer_obj = POS_probe_trainer(
            probe, device, num_shift_pos=num_shift_positions
        )
    else:
        probe_trainer_obj = probe_trainer(
            probe,
            device,
            dataset_name,
            feature_truncation_method=feature_truncation_method,
        )

    avg_test_accuracy = probe_trainer_obj.test_probe(
        probe_batch_size, num_test_subsets, load_model=True, learning_rate=learning_rate
    )

    print(f"probe test accuracy: {avg_test_accuracy:.2f}")

    return probe_trainer_obj, avg_test_accuracy


def get_probe_input_feature_size(
    probing_dataset_creator_obj, truncation_method, truncated_feature_size=None
):
    num_training_examples = probing_dataset_creator_obj.num_training_examples
    probe_input_example_size = probing_dataset_creator_obj.probe_input_example_size

    if (
        num_training_examples / 10
    ) >= probe_input_example_size or truncation_method is None:
        return probe_input_example_size
    else:
        if truncated_feature_size is None or truncated_feature_size == 0:
            new_input_feature_size = int(num_training_examples / 10)
            if (
                "average" in truncation_method or "avg" in truncation_method
            ) and probe_input_example_size % new_input_feature_size != 0:
                # new_input_feature_size -= probe_input_example_size % new_input_feature_size
                # Find all factors of probe_input_example_size
                factors_of_probe_input_example_size = [
                    i
                    for i in range(1, probe_input_example_size + 1)
                    if probe_input_example_size % i == 0
                ]
                # Filter factors to find those that are smaller than new_input_feature_size
                valid_factors = [
                    factor
                    for factor in factors_of_probe_input_example_size
                    if factor < new_input_feature_size
                ]
                new_input_feature_size = max(valid_factors)
        else:
            new_input_feature_size = truncated_feature_size

        return new_input_feature_size
