import numpy as np
import torch

from src.Probing.POSProbing.constants import early_stopping
from src.Probing.POSProbing.constants import EPOCHS
from src.Probing.POSProbing.constants import extracted_layers
from src.Probing.POSProbing.constants import learning_rate
from src.Probing.POSProbing.constants import LM_batch_size
from src.Probing.POSProbing.constants import LM_name
from src.Probing.POSProbing.constants import max_sent_len
from src.Probing.POSProbing.constants import num_shift_positions
from src.Probing.POSProbing.constants import num_test_subsets
from src.Probing.POSProbing.constants import num_train_subsets
from src.Probing.POSProbing.constants import num_validation_subsets
from src.Probing.POSProbing.constants import POS_path
from src.Probing.POSProbing.constants import probe_batch_size
from src.Probing.POSProbing.constants import sentence_path
from src.Probing.POSProbing.constants import test_split
from src.Probing.POSProbing.constants import train_split
from src.Probing.POSProbing.constants import validation_split
from src.Probing.probing_functions import create_POS_probing_dataset
from src.Probing.probing_functions import train_probe

# from src.Probing.probing_functions import test_probe


# set seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)
# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create POS dataset
probe_input_example_size, num_labels = create_POS_probing_dataset(
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
)
##################################################################

# train POS probe
POS_probe_trainer = train_probe(
    "POS",
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
    num_shift_positions,
)

# POS_probe_trainer = test_probe(
#     "POS",
#     probe_input_example_size,
#     num_labels,
#     device,
#     probe_batch_size,
#     num_test_subsets,
#     num_shift_positions,
#     learning_rate=learning_rate,
# )
