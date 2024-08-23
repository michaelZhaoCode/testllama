import gc
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from old_code.Create_Probe_POS_dataset_from_LM import count_POS_tag_occurrence
from old_code.Create_Probe_POS_dataset_from_LM import (
    create_full_POS_conversion_dictionary,
)
from old_code.Create_Probe_POS_dataset_from_LM import create_Probe_dataset
from old_code.Create_Probe_POS_dataset_from_LM import get_POS_occurrence_prob
from old_code.Create_Probe_POS_dataset_from_LM import load_POS_dataset_csv
from old_code.Create_Probe_POS_dataset_from_LM import remove_long_sentences
from old_code.Create_Probe_POS_dataset_from_LM import split_dataset_subsets
from old_code.linear_probe_training import linear_probe
from old_code.linear_probe_training import model_fit
from old_code.linear_probe_training import model_test
from old_code.probe_experiment_variables import early_stopping
from old_code.probe_experiment_variables import epochs
from old_code.probe_experiment_variables import label_shift as assigned_label_shift
from old_code.probe_experiment_variables import learning_rate
from old_code.probe_experiment_variables import LM_batch_size
from old_code.probe_experiment_variables import LM_embedding_size
from old_code.probe_experiment_variables import LM_name
from old_code.probe_experiment_variables import LM_num_layers
from old_code.probe_experiment_variables import LM_probed_layers
from old_code.probe_experiment_variables import num_test_subsets
from old_code.probe_experiment_variables import num_train_subsets
from old_code.probe_experiment_variables import num_validation_subsets
from old_code.probe_experiment_variables import probe_batch_size
from old_code.probe_experiment_variables import randomize_labels
from old_code.probe_experiment_variables import sentence_len_cap
from old_code.probe_experiment_variables import shift_direction
from old_code.probe_experiment_variables import test_only
from old_code.probe_experiment_variables import test_split
from old_code.probe_experiment_variables import train_split
from old_code.probe_experiment_variables import use_control_dataset
from old_code.probe_experiment_variables import validation_split

# set seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

assert (
    train_split + validation_split + test_split == 1
), "train-validation-test split doesn't add to 100%"

assert "bert" in LM_name or "gpt2" in LM_name, "used model isn't a BERT or GPT2 model"

# # GPT2 models are autoregressive models, so this experiment works only if the shifting is to the right
# if "gpt2" in LM_name:
#     assert shift_direction == "right", "chosen shift direction doesn't work with GPT2 models"

if LM_probed_layers is not None and LM_probed_layers != []:
    for layer_num in LM_probed_layers:
        assert (
            layer_num >= 0 and layer_num < LM_num_layers
        ), "layer number not within range of LM layers"


# Probe Dataset Creation

# first load the POS dataset from the penn treebank files & create the POS conversion dictionary
# save all of these variables for later use
# sent_list, POS_list = load_POS_dataset()  # load the penn treebank dataset
# POS_dict = create_full_POS_conversion_dictionary(POS_list)

start_time = time.time()
# sent_list, POS_list = load_POS_dataset()       # load the penn treebank dataset
sent_list, POS_list = load_POS_dataset_csv()
# POS_conversion_dict = load_POS_conversion_dictionary() # dictionary that converts each POS tag into a class value
POS_conversion_dict = create_full_POS_conversion_dictionary(
    POS_list
)  # dictionary that converts each POS tag into a class value

POS_occurrence_probs = get_POS_occurrence_prob(
    POS_list
)  # dictionary that has each POS tag along with its occurrence probability

sent_list, POS_list = remove_long_sentences(sentence_len_cap, sent_list, POS_list)

count_POS_tag_occurrence(POS_list)

# split dataset into training, validation, and test sets
train_sent_list, temp_sent_list, train_POS_list, temp_POS_list = train_test_split(
    sent_list, POS_list, train_size=train_split, shuffle=True, random_state=1
)
(
    validation_sent_list,
    test_sent_list,
    validation_POS_list,
    test_POS_list,
) = train_test_split(
    temp_sent_list,
    temp_POS_list,
    train_size=validation_split / (validation_split + test_split),
    shuffle=True,
    random_state=1,
)

# split each set into smaller subsets
train_sent_subsets, train_POS_subsets = split_dataset_subsets(
    train_sent_list, train_POS_list, num_train_subsets
)
validation_sent_subsets, validation_POS_subsets = split_dataset_subsets(
    validation_sent_list, validation_POS_list, num_validation_subsets
)
test_sent_subsets, test_POS_subsets = split_dataset_subsets(
    test_sent_list, test_POS_list, num_test_subsets
)

# for label_shift in range(1, 3):
for label_shift in range(1):
    label_shift = assigned_label_shift
    print(f"currently using shift = {label_shift}")

    probe_dataset_execution_time = time.time()

    if test_only is False:
        print("creating training dataset")
        # extract hidden states for training set
        for i in range(len(train_sent_subsets)):
            create_Probe_dataset(
                train_sent_subsets[i],
                train_POS_subsets[i],
                POS_conversion_dict,
                LM_batch_size,
                device,
                set_name="train",
                subset_num=i,
                label_shift=label_shift,
                shift_direction=shift_direction,
                LM_embedding_size=LM_embedding_size,
                LM_num_layers=LM_num_layers,
                LM_probed_layers=LM_probed_layers,
                use_control_dataset=use_control_dataset,
                LM_name=LM_name,
                POS_occurrence_probs=POS_occurrence_probs,
                randomize_labels=randomize_labels,
            )

        print("creating validation dataset")
        # extract hidden states for validation set
        for i in range(len(validation_sent_subsets)):
            create_Probe_dataset(
                validation_sent_subsets[i],
                validation_POS_subsets[i],
                POS_conversion_dict,
                LM_batch_size,
                device,
                set_name="validation",
                subset_num=i,
                label_shift=label_shift,
                shift_direction=shift_direction,
                LM_embedding_size=LM_embedding_size,
                LM_num_layers=LM_num_layers,
                LM_probed_layers=LM_probed_layers,
                use_control_dataset=use_control_dataset,
                LM_name=LM_name,
                POS_occurrence_probs=POS_occurrence_probs,
                randomize_labels=randomize_labels,
            )

    print("creating test dataset")
    # extract hidden states for test set
    for i in range(len(test_sent_subsets)):
        create_Probe_dataset(
            test_sent_subsets[i],
            test_POS_subsets[i],
            POS_conversion_dict,
            LM_batch_size,
            device,
            set_name="test",
            subset_num=i,
            label_shift=label_shift,
            shift_direction=shift_direction,
            LM_embedding_size=LM_embedding_size,
            LM_num_layers=LM_num_layers,
            LM_probed_layers=LM_probed_layers,
            use_control_dataset=use_control_dataset,
            LM_name=LM_name,
            POS_occurrence_probs=POS_occurrence_probs,
            randomize_labels=randomize_labels,
        )

    print(
        f"time taken to create probing dataset: {(time.time() - probe_dataset_execution_time) / 60:.3f} minutes"
    )

    ########################################################################################################

    print("\n\n\n\n\n")

    # Probe Training

    # compute input and output dimensions of the probe
    if LM_probed_layers is None or LM_probed_layers == []:
        probe_input_dim = LM_num_layers * LM_embedding_size
    else:
        probe_input_dim = len(LM_probed_layers) * LM_embedding_size
    probe_output_dim = len(POS_conversion_dict)

    print(f"number of classes for the current experiment: {probe_output_dim}")

    # instantiate probing model
    probing_model = linear_probe(probe_input_dim, probe_output_dim).to(device)

    if test_only is False:
        training_time = time.time()
        # train the model
        model_fit(
            probing_model,
            device,
            batch_size=probe_batch_size,
            learning_rate=learning_rate,
            EPOCHS=epochs,
            early_stopping=early_stopping,
            num_train_subsets=num_train_subsets,
            num_validation_subsets=num_validation_subsets,
            label_shift=label_shift,
            shift_direction=shift_direction,
        )
        print(f"training time: {(time.time() - training_time) / 60:.3f} minutes")

        # save model
        model_file_path = (
            os.getcwd()
            + os.path.sep
            + "old_code"
            + os.path.sep
            + "data"
            + os.path.sep
            + f"probe_model_{shift_direction}_{label_shift}.pt"
        )
        torch.save(probing_model.state_dict(), model_file_path)
    else:
        # Load trained model
        model_file_path = os.path.join(
            os.getcwd(),
            "old_code",
            "data",
            f"probe_model_{shift_direction}_{label_shift}.pt",
        )
        probing_model.load_state_dict(
            torch.load(model_file_path, map_location=torch.device(device))
        )

    # test the trained model
    test_accuracy, _ = model_test(
        probing_model,
        device,
        batch_size=probe_batch_size,
        num_subsets=num_test_subsets,
        label_shift=label_shift,
        shift_direction=shift_direction,
    )
    print(f"Test accuracy is {test_accuracy:.3f}")

    print("\n\n\n")

    # for the control test
    if label_shift != 0 and use_control_dataset is True:
        print("starting training and testing for the control model")
        control_probing_model = linear_probe(probe_input_dim, probe_output_dim).to(
            device
        )

        training_time = time.time()
        # train the model
        model_fit(
            control_probing_model,
            device,
            batch_size=probe_batch_size,
            learning_rate=learning_rate,
            EPOCHS=epochs,
            early_stopping=early_stopping,
            num_train_subsets=num_train_subsets,
            num_validation_subsets=num_validation_subsets,
            label_shift=label_shift,
            shift_direction=shift_direction,
            control_test=True,
        )

        # test the trained model
        test_accuracy, *_ = model_test(
            control_probing_model,
            device,
            batch_size=probe_batch_size,
            num_subsets=num_test_subsets,
            label_shift=label_shift,
            shift_direction=shift_direction,
            control_test=True,
        )
        print(f"control Test accuracy is {test_accuracy:.3f}")
        print(f"training time: {(time.time() - training_time) / 60:.3f} minutes")

        # save model
        model_file_path = (
            os.getcwd()
            + os.path.sep
            + "data"
            + os.path.sep
            + f"control_probe_model_{shift_direction}_{label_shift}.pt"
        )
        torch.save(control_probing_model.state_dict(), model_file_path)

    ########################################################################################################

    # Delete datasets
    # dataset directory
    dataset_dir = os.getcwd() + os.path.sep + "data"
    output_datasets_keyword = "output"
    input_datasets_keyword = "input"

    # List all files in the specified directory
    dir_files = os.listdir(dataset_dir)

    # Filter files based on the keyword in their names
    filtered_files = [
        file
        for file in dir_files
        if ((output_datasets_keyword in file) or (input_datasets_keyword in file))
        and ".csv" not in file
    ]

    for file in filtered_files:
        file_path = os.path.join(dataset_dir, file)
        try:
            # Delete the file
            os.remove(file_path)
            print(f"File {file_path} deleted successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    ########################################################################################################
    print("\n\n\n\n\n\n\n\n")


# this is to pause the program when debugging to see the plots, otherwise, they will close once the code finishes
x = 0
