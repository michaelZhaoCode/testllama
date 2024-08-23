import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from old_code.Create_Probe_POS_dataset_from_LM import load_probe_dataset

# from Create_Probe_POS_dataset_BERT import num_train_subsets, num_validation_subsets, num_test_subsets


class linear_probe(torch.nn.Module):
    def __init__(self, LM_num_neurons, num_output_classes):
        super().__init__()

        self.LM_num_neurons = LM_num_neurons
        self.num_output_classes = num_output_classes

        self.linear_NN = torch.nn.Linear(
            self.LM_num_neurons, self.num_output_classes, bias=False
        )  # fully connected layer

    def forward(self, x):
        """
        x: torch.tensor of shape (batch_size, LM_num_neurons)
        """

        linear_output = self.linear_NN(x)

        return linear_output


class probe_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        device,
        set_name,
        subset_num=0,
        label_shift=0,
        shift_direction="left",
        control_test=False,
    ):
        X, Y = load_probe_dataset(
            device,
            set_name,
            subset_num,
            label_shift=label_shift,
            shift_direction=shift_direction,
            control_test=control_test,
        )

        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def prediction_accuracy(predicted, ground_truth, softmax_fn):
    """
    Description:
        this function takes the linear output of the NN and applies the softmax and argmax function,
        then it computes the accuracy

    Parameters:
        predicted: torch tensor, predicted output from the model
        ground_truth: torch tensor, dataset labels
        softmax_fn: torch.nn.softmax function

    Returns:
        average_accuracy_per_batch: float, batch accuracy computed between the predicted and ground truth values
    """

    batch_size = ground_truth.shape[0]

    softmax_output = softmax_fn(predicted)
    predicted_class = torch.argmax(softmax_output, dim=-1)

    accuracy_counter = 0
    for i in range(batch_size):
        if predicted_class[i] == ground_truth[i]:
            accuracy_counter += 1

    average_accuracy_per_batch = accuracy_counter / batch_size

    return average_accuracy_per_batch


def model_fit(
    NN_model,
    device,
    batch_size=512,
    learning_rate=0.001,
    EPOCHS=10,
    early_stopping=True,
    num_train_subsets=4,
    num_validation_subsets=1,
    label_shift=0,
    shift_direction="left",
    control_test=False,
):
    """
    Description:
        this function trains the model

    Parameters:
        NN_model: model that is intended to be trained
        device: device which is used for computation (CPU/GPU/TPU)
        batch_size: int, size of the batches of sentences processed at a time
        learning_rate: float
        EPOCHS: int, number of passes through whole dataset
        early_stopping: bool, determines whether or not early stopping should be applied
        num_train_subsets: int, number of subsets for training set
        num_validation_subsets: int, number of subsets for validation set
        label_shift: int, denotes number of positions by which the POS label is shifted
        shift_direction: str, denotes the direction to which the POS labels should be shifted (either "left" or "right")
        control_test: boolean, chooses whether to load the output for the shifted dataset or for the control dataset

    Returns:
        None
    """

    # set seeds for reproducability
    np.random.seed(1)
    torch.manual_seed(1)

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(NN_model.parameters(), lr=learning_rate)

    # these 2 variables are used for plotting loss over epochs
    training_loss_per_epoch = np.zeros(EPOCHS)
    validation_loss_per_epoch = np.zeros(EPOCHS)
    # these 2 variables are used for plotting accuracy over epochs
    training_accuracy_per_epoch = np.zeros(EPOCHS)
    validation_accuracy_per_epoch = np.zeros(EPOCHS)

    # used for getting prediction accuracy
    prediction_softmax_fn = torch.nn.Softmax(dim=-1)

    # Set up early stopping parameters
    patience = 3  # Number of epochs to wait for improvement
    early_stopping_counter = 0
    best_validation_loss = float("inf")  # Initialize with a large value
    executed_epochs = EPOCHS

    # training loop
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        # training loss and accuracy calculated over a whole epoch
        running_train_loss = 0.0
        running_train_accuracy = 0.0

        # for loop used because training set is split into subsets
        num_train_batches = 0
        for i in range(num_train_subsets):
            print(f"currently using training subset number: {i}")
            train_data = probe_dataset(
                device,
                set_name="train",
                subset_num=i,
                label_shift=label_shift,
                shift_direction=shift_direction,
                control_test=control_test,
            )
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_data, batch_size=batch_size, shuffle=True
            )
            num_train_batches += len(train_dataloader)

            for batch in train_dataloader:
                inputs = batch[0]
                labels = batch[1]

                # zero gradients for each batch
                optimizer.zero_grad()

                # Make predictions for this batch
                predicted_outputs = NN_model.forward(inputs)

                # Compute the loss and its gradients
                train_loss = loss_fn(predicted_outputs, labels)
                train_loss.backward()

                # Adjust learning weights
                optimizer.step()

                running_train_loss += train_loss

                train_accuracy = prediction_accuracy(
                    predicted_outputs, labels, prediction_softmax_fn
                )
                running_train_accuracy += train_accuracy

            del predicted_outputs
            del batch
            del train_data
            del train_dataloader
            gc.collect()
            torch.cuda.empty_cache()

        avg_train_loss = running_train_loss / num_train_batches
        training_loss_per_epoch[epoch] = avg_train_loss

        avg_train_accuracy = running_train_accuracy / num_train_batches
        training_accuracy_per_epoch[epoch] = avg_train_accuracy

        avg_validation_accuracy, avg_validation_loss = model_test(
            NN_model,
            device,
            batch_size=batch_size,
            loss_fn=loss_fn,
            set_name="validation",
            num_subsets=num_validation_subsets,
            label_shift=label_shift,
            shift_direction=shift_direction,
            control_test=control_test,
        )

        validation_loss_per_epoch[epoch] = avg_validation_loss
        validation_accuracy_per_epoch[epoch] = avg_validation_accuracy

        print(
            f"EPOCH: {epoch} \t LOSS train {avg_train_loss:.3f}, \t valid {avg_validation_loss:.3f}, "
            f"ACC train {avg_train_accuracy:.3f}, \t valid {avg_validation_accuracy:.3f}, "
            f"epoch time: {(time.time()-epoch_start_time)/60:.3f} minutes"
        )

        if early_stopping is True:
            # Check if the validation loss has improved
            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Check if early stopping criteria are met
            if early_stopping_counter >= patience:
                print(f"Early stopping after {epoch} epochs.")
                executed_epochs = epoch + 1
                break

    if early_stopping is True:
        plot_x_axis = np.arange(executed_epochs)

        # remove dimensions with zeros
        training_loss_per_epoch = training_loss_per_epoch[training_loss_per_epoch != 0]
        validation_loss_per_epoch = validation_loss_per_epoch[
            validation_loss_per_epoch != 0
        ]
        training_accuracy_per_epoch = training_accuracy_per_epoch[
            training_accuracy_per_epoch != 0
        ]
        validation_accuracy_per_epoch = validation_accuracy_per_epoch[
            validation_accuracy_per_epoch != 0
        ]

    else:
        plot_x_axis = np.arange(EPOCHS)

    plot_2_curves(
        plot_x_axis,
        training_loss_per_epoch,
        validation_loss_per_epoch,
        label1="train loss",
        label2="validation loss",
        title="Training & validation losses",
    )

    plot_2_curves(
        plot_x_axis,
        training_accuracy_per_epoch,
        validation_accuracy_per_epoch,
        label1="train accuracy",
        label2="validation accuracy",
        title="Training & validation accuracies",
    )

    model_file_path = (
        os.getcwd()
        + os.path.sep
        + "old_code"
        + os.path.sep
        + "data"
        + os.path.sep
        + "probe_model.pt"
    )
    torch.save(NN_model.state_dict(), model_file_path)

    return


def model_test(
    NN_model,
    device,
    batch_size=512,
    loss_fn=None,
    set_name="test",
    num_subsets=0,
    label_shift=0,
    shift_direction="left",
    control_test=False,
):
    """
    Description:
        this function tests the model

    Parameters:
        NN_model: model that is intended to be trained
        device: device which is used for computation (CPU/GPU/TPU)
        batch_size: int, size of the batches of sentences processed at a time
        loss_fn: loss function used for training. Makes sense to be used for validation set and not test set
        set_name: str, dataset with which model is tested. Either takes the value "validation" or "test"
        num_subsets: int, number of subsets for input dataset
        label_shift: int, denotes number of positions by which the POS label is shifted
        shift_direction: str, denotes the direction to which the POS labels should be shifted (either "left" or "right")
        control_test: boolean, chooses whether to load the output for the shifted dataset or for the control dataset

    Returns:
        avg_loss: float, loss computed between predicted outputs and ground truth values (for whole dataset)
        avg_accuracy: float, accuracy computed between predicted outputs and ground truth values (for whole dataset)
    """

    assert (
        set_name == "test" or set_name == "validation"
    ), '"set_name" is assigned an invalid string'

    # used for getting prediction accuracy
    prediction_softmax_fn = torch.nn.Softmax(dim=-1)

    running_loss = 0.0
    running_accuracy = 0.0

    # for loop used because input set is split into subsets
    num_batches = 0
    for i in range(num_subsets):
        print(f"currently using {set_name} subset number: {i}")
        data = probe_dataset(
            device,
            set_name=set_name,
            subset_num=i,
            label_shift=label_shift,
            shift_direction=shift_direction,
            control_test=control_test,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=data, batch_size=batch_size, shuffle=True
        )

        num_batches += len(dataloader)

        for batch in dataloader:
            inputs = batch[0]
            labels = batch[1]

            with torch.no_grad():
                # Make predictions for this batch
                NN_model.eval()
                predicted_outputs = NN_model.forward(inputs)

            # Compute the loss
            if loss_fn is None:
                loss = 0
            else:
                loss = loss_fn(predicted_outputs, labels)

            running_loss += loss

            accuracy = prediction_accuracy(
                predicted_outputs, labels, prediction_softmax_fn
            )
            running_accuracy += accuracy

        del predicted_outputs
        del batch
        del data
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()

    # average loss and accuracy calculated over the whole dataset
    avg_loss = running_loss / num_batches
    avg_accuracy = running_accuracy / num_batches

    return avg_accuracy, avg_loss


def plot_2_curves(x, y1, y2, label1, label2, title):
    """
    Description:
        function that plots 2 curves on the same graph

    Parameters:
        x: input array
        y1: first output array
        y2: second output array
        label1: str, label for first output array
        label2: str, label for second output array
        title: str, title for the plot

    Returns:
        None
    """
    plt.figure()
    plt.plot(x, y1, color="b", label=label1)
    plt.plot(x, y2, color="r", label=label2)
    plt.title(title)
    plt.legend()
    plt.show(block=False)

    return


def get_num_dataset_labels(device):
    """
    Description:
        gets the number of labels in the dataset for determining the size of output dimension of the probe

    Parameters:
        device: device which is used for computation (CPU/GPU/TPU)

    Returns:
        num_unique_labels: int, number of labels in the dataset
    """

    # dataset directory
    dataset_dir = os.getcwd() + os.path.sep + "data"
    output_datasets_keyword = "output"

    # List all files in the specified directory
    dir_files = os.listdir(dataset_dir)

    # Filter files based on the keyword in their names
    filtered_files = [
        file
        for file in dir_files
        if output_datasets_keyword in file and ".csv" not in file
    ]

    # tensor containing all of the unique labels
    unique_labels = torch.empty(0)
    # Read the content of each filtered file
    for file_name in filtered_files:
        file_path = os.path.join(dataset_dir, file_name)
        output_data = torch.load(file_path, map_location=torch.device(device))

        # get unique labels in the read file and append that to the unique_labels variable
        file_unique_labels = torch.unique(output_data)
        concatenated_labels = torch.cat((unique_labels, file_unique_labels))
        unique_labels = torch.unique(concatenated_labels)

        del concatenated_labels
        del file_unique_labels
        del output_data
        gc.collect()
        torch.cuda.empty_cache()

    num_unique_labels = unique_labels.shape[0]  # number of labels in the dataset

    del unique_labels
    gc.collect()
    torch.cuda.empty_cache()

    return num_unique_labels
