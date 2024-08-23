import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.Probing.constants import dataset_name
from src.Probing.constants import dataset_path
from src.Probing.constants import early_stopping
from src.Probing.constants import EPOCHS
from src.Probing.constants import extracted_layers
from src.Probing.constants import HF_TOKEN
from src.Probing.constants import learning_rate
from src.Probing.constants import LM_batch_size
from src.Probing.constants import num_output_labels
from src.Probing.constants import num_test_subsets
from src.Probing.constants import num_train_subsets
from src.Probing.constants import num_validation_subsets
from src.Probing.constants import probe_batch_size
from src.Probing.constants import probed_LM
from src.Probing.constants import test_split
from src.Probing.constants import train_split
from src.Probing.constants import truncated_feature_size
from src.Probing.constants import truncation_method
from src.Probing.constants import validation_split
from src.Probing.probing_functions import create_sent_probing_dataset
from src.Probing.probing_functions import get_probe_input_feature_size
from src.Probing.probing_functions import train_probe

# from src.Probing.probing_functions import test_probe

import torch.distributed as dist

# Initialize the process group (this needs to be called on all nodes)
dist.init_process_group(backend='nccl')

# Get the rank of the process (which node/GPU this is)
rank = dist.get_rank()

# Assign the correct GPU based on rank
device = torch.device(f'cuda:{rank}')



MAX_LAYER_LOOP = 1


# set seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)
# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_time = time.time()
if extracted_layers:
    # create POS dataset
    probing_dataset_creator_obj = create_sent_probing_dataset(
        probed_LM,
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
        HF_TOKEN,
    )
    ##################################################################

    # get probe input feature size
    probe_input_feature_size = get_probe_input_feature_size(
        probing_dataset_creator_obj, truncation_method, truncated_feature_size
    )

    # get probing data save path
    probe_data_save_dir = probing_dataset_creator_obj.get_probe_data_save_dir()

    # train probe
    probe_trainer_obj, _ = train_probe(
        dataset_name,
        probe_input_feature_size,
        num_output_labels,
        device,
        probe_batch_size,
        learning_rate,
        EPOCHS,
        early_stopping,
        num_train_subsets,
        num_validation_subsets,
        num_test_subsets,
        probe_data_save_dir=probe_data_save_dir,
        feature_truncation_method=truncation_method,
    )

# Extracted layers is empty so loop
else:
    # Loop through layers if no specific layers are extracted initially
    layer_nums = []
    test_accuracies = []
    probing_dataset_creator_obj = None

    for i in range(MAX_LAYER_LOOP + 1):
        print(f"Running LM layer {i}")
        extracted_layers = [i]

        # Create dataset and probing object
        probing_dataset_creator_obj = create_sent_probing_dataset(
            probed_LM,
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
            HF_TOKEN,
            probing_dataset_creator_obj=probing_dataset_creator_obj,  # Reuse existing object
        )

        # Get probe input feature size
        probe_input_feature_size = get_probe_input_feature_size(
            probing_dataset_creator_obj, truncation_method, truncated_feature_size
        )

        # Get probing data save path
        probe_data_save_dir = probing_dataset_creator_obj.get_probe_data_save_dir()

        # Train probe on the extracted features
        probe_trainer_obj, avg_test_accuracy = train_probe(
            dataset_name,
            probe_input_feature_size,
            num_output_labels,
            device,  # We use a single device for the probe training
            probe_batch_size,
            learning_rate,
            EPOCHS,
            early_stopping,
            num_train_subsets,
            num_validation_subsets,
            num_test_subsets,
            probe_data_save_dir=probe_data_save_dir,
            feature_truncation_method=truncation_method,
        )

        layer_nums.append(i)
        test_accuracies.append(avg_test_accuracy)

    # Plot the results
    plt.scatter(layer_nums, test_accuracies)
    plt.plot(layer_nums, test_accuracies)
    plt.title("Average Test Accuracy vs Probe Layer")
    plt.xlabel("Probe Layer")
    plt.ylabel("Test Accuracy")

    # Save the plot results
    full_path = os.path.join(os.getcwd(), "data", "probe_results", probed_LM)
    os.makedirs(full_path, exist_ok=True)

    if truncation_method is not None:
        file_path = os.path.join(full_path, f"{truncation_method}_{probe_input_feature_size}.png")
    else:
        file_path = os.path.join(full_path, f"NoTruncation_{probe_input_feature_size}.png")

    plt.savefig(file_path)

# Report the total time taken
print(f"\nTime taken for experiment: {(time.time() - start_time) / 60:.2f} minutes")
if truncation_method is not None:
    print(f"Experiment conducted for {truncation_method} with feature vector size: {probe_input_feature_size}")

# probe_trainer_obj = test_probe(
#     dataset_name,
#     probe_input_example_size,
#     num_output_labels,
#     device,
#     probe_batch_size,
#     num_test_subsets,
#     learning_rate=learning_rate,
#     feature_truncation_method=truncation_method,
# )
