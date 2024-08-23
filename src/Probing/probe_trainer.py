import os
import time

import numpy as np
import torch

from src.Probing.probe import linear_probe


class probe_trainer:
    def __init__(
        self,
        model: linear_probe,
        device,
        dataset_name,
        dataset_dir=None,
        feature_truncation_method="average",
    ):
        self.model = model
        self.device = device
        self.dataset_name = dataset_name
        self.dataset_dir = None if dataset_dir is None else dataset_dir
        # training hyperparameters
        self.loss_fn = None
        self.optimizer = None
        self.learning_rate = None
        self.batch_size = None
        # early stopping attributes
        self.early_stopping_patience = None
        self.early_stopping_counter = None
        self.best_validation_loss = None
        # plotting arrays
        self.train_loss_per_epoch = None
        self.train_acc_per_epoch = None
        self.validation_loss_per_epoch = None
        self.validation_acc_per_epoch = None
        # temporary variables for training, validation, and testing
        self.num_batches = None
        self.running_loss = None
        self.running_acc = None
        self.feature_truncation_method = feature_truncation_method
        self.data_truncator_obj = None

    def get_loss_fn(self):
        if self.model.output_dim == 1:
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.CrossEntropyLoss()

    def init_early_stopping(self, early_stopping_patience=3):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.best_validation_loss = float("inf")  # Initialize with a large value
        return

    def init_training_variables(
        self, EPOCHS, learning_rate, batch_size, early_stopping, early_stopping_patience
    ):
        # batch size
        self.batch_size = batch_size
        # learning rate
        self.learning_rate = learning_rate
        # loss function
        self.loss_fn = self.get_loss_fn()
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # early stopping
        self.init_early_stopping(
            early_stopping_patience
        ) if early_stopping is True else None
        # variables for plotting loss and accuracy over epochs
        self.train_loss_per_epoch, self.validation_loss_per_epoch = np.zeros(
            EPOCHS
        ), np.zeros(EPOCHS)
        self.train_acc_per_epoch, self.validation_acc_per_epoch = np.zeros(
            EPOCHS
        ), np.zeros(EPOCHS)
        return

    def load_dataset(self, dataset_purpose, subset_num):
        dataset = probing_dataset(
            self.device,
            dataset_purpose=dataset_purpose,
            subset_num=subset_num,
            dataset_dir=self.dataset_dir,
        )

        # truncate input data
        original_input_data = dataset.get_input_data()
        updated_input_data = self.truncate_features(
            original_input_data, dataset_purpose
        )
        dataset.update_input_data(updated_input_data)
        ################################

        return dataset

    def truncate_features(self, original_input, dataset_purpose=None):
        """
        Description:
            truncate the feature size of the input examples to match the input probe dimensions

        Parameters:
            original_input : torch tensor, tensor containing the probing original input data
            dataset_purpose: str, "training"/"validation"/"test"

        Returns:
            updated_input: torch tensor, truncated input data tensor
        """
        original_feature_size = original_input.size(-1)
        new_feature_size = self.model.input_dim

        if original_feature_size == new_feature_size:
            return original_input
        else:
            if "train" in dataset_purpose:
                self.data_truncator_obj = data_truncator(
                    self.feature_truncation_method,
                    original_feature_size,
                    new_feature_size,
                )
            updated_input = self.data_truncator_obj.truncate_data(
                original_input, dataset_purpose
            )
            return updated_input

    def load_dataset_to_dataloader(self, batch_size, dataset_purpose, subset_num=0):
        dataset = self.load_dataset(dataset_purpose, subset_num)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )
        return dataloader

    def early_get_dataloader(self, num_subsets, dataset_purpose: str, batch_size: int):
        if num_subsets == 1:
            return self.load_dataset_to_dataloader(batch_size, dataset_purpose)
        else:
            return None

    def check_early_stopping(self, avg_validation_loss):
        # Check if the validation loss has improved
        if avg_validation_loss < self.best_validation_loss:
            self.best_validation_loss = avg_validation_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        # Check if early stopping criteria are met
        if self.early_stopping_counter >= self.early_stopping_patience:
            stop = True
        else:
            stop = False

        return stop

    def train(
        self,
        batch_size,
        learning_rate,
        EPOCHS,
        early_stopping=False,
        early_stopping_patience=3,
        num_train_subsets=1,
        num_validation_subsets=1,
    ):
        """
        Description:
            this function trains the probe

        Parameters:
            batch_size: int, size of the batches of sentences processed at a time
            learning_rate: float, learning rate for training
            EPOCHS: int, number of passes through whole dataset
            early_stopping: bool, determines whether to do early stopping
            early_stopping_patience: int, determines number of epochs to wait before stopping training in case
                                        validation accuracy doesn't change
            num_train_subsets: int, number of subsets for training set
            num_validation_subsets: int, number of subsets for validation set

        Returns:
            None
        """

        # set seeds for reproducability
        np.random.seed(1)
        torch.manual_seed(1)
        # initialize training variables
        self.init_training_variables(
            EPOCHS, learning_rate, batch_size, early_stopping, early_stopping_patience
        )
        # train and validation dataloaders
        train_dataloader = self.early_get_dataloader(
            num_train_subsets, "training", batch_size
        )
        validation_dataloader = self.early_get_dataloader(
            num_validation_subsets, "validation", batch_size
        )
        # training loop
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            avg_train_loss, avg_train_accuracy = self.train_epoch(
                epoch, batch_size, train_dataloader, num_train_subsets
            )
            avg_validation_loss, avg_validation_accuracy = self.test(
                batch_size,
                validation_dataloader,
                num_validation_subsets,
                dataset_purpose="validation",
            )
            self.validation_loss_per_epoch[epoch] = avg_validation_loss
            self.validation_acc_per_epoch[epoch] = avg_validation_accuracy
            print(
                f"EPOCH: {epoch} \t LOSS train {avg_train_loss:.2f}, \t valid {avg_validation_loss:.2f}, "
                f"ACC train {avg_train_accuracy:.2f}, \t valid {avg_validation_accuracy:.2f}, "
                f"epoch time: {(time.time() - epoch_start_time) / 60:.2f} minutes"
            )
            # check for early stopping
            if early_stopping is True:
                stop = self.check_early_stopping(avg_validation_loss)
                if stop is True:
                    print(f"Early stopping after {epoch+1} epochs.")
                    break
        # plot training and validation results
        self.save_model()
        return

    def train_epoch(self, epoch_num, batch_size, train_dataloader, num_train_subsets):
        # training loss and accuracy calculated over a whole epoch
        self.running_loss, self.running_acc = 0.0, 0.0
        # for loop used because training set is split into subsets
        self.num_batches = 0
        for i in range(num_train_subsets):
            self.run_subset(batch_size, train_dataloader, i, dataset_purpose="training")

        # modify plotting variables with the new epoch results
        avg_train_loss = self.running_loss / self.num_batches
        avg_train_accuracy = self.running_acc / self.num_batches
        self.train_loss_per_epoch[epoch_num] = avg_train_loss
        self.train_acc_per_epoch[epoch_num] = avg_train_accuracy

        return avg_train_loss, avg_train_accuracy

    def run_subset(self, batch_size, dataloader, subset_num, dataset_purpose):
        print(f"currently using {dataset_purpose} subset number: {subset_num}")
        if dataloader is None:
            dataloader = self.load_dataset_to_dataloader(
                batch_size, dataset_purpose, subset_num=subset_num
            )
        else:
            None

        self.num_batches += len(dataloader)
        for batch in dataloader:
            if "train" in dataset_purpose:
                self.train_batch(batch)
            else:
                self.test_batch(batch)

        return

    def get_avg_batch_acc(self, predicted_logits, ground_truth):
        batch_size = ground_truth.shape[0]

        if self.model.output_dim == 1:
            threshold = 0.5
            predicted_classes = (predicted_logits >= threshold).int()
        else:
            predicted_classes = torch.argmax(predicted_logits, dim=-1)

        # Element-wise comparison
        comparison = predicted_classes == ground_truth

        # Count the number of equal elements
        num_equal_elements = comparison.sum().item()

        avg_batch_acc = num_equal_elements / batch_size

        return avg_batch_acc

    def train_batch(self, batch):
        inputs = batch[0]
        labels = torch.squeeze(batch[1])
        # zero gradients for each batch
        self.optimizer.zero_grad()
        # Make predictions for this batch
        predicted_logits = torch.squeeze(self.model.forward(inputs))
        # Compute the loss and its gradients
        train_loss = self.loss_fn(predicted_logits, labels)
        train_loss.backward()
        # Adjust learning weights
        self.optimizer.step()
        # adjust running loss & accuracy
        self.running_loss += train_loss
        train_accuracy = self.get_avg_batch_acc(predicted_logits, labels)
        self.running_acc += train_accuracy

        return

    def test_batch(self, batch):
        inputs = batch[0]
        labels = torch.squeeze(batch[1])
        # Make predictions for this batch
        with torch.no_grad():
            self.model.eval()
            predicted_logits = torch.squeeze(self.model.forward(inputs))
        # Compute the loss
        loss = self.loss_fn(predicted_logits, labels)
        # adjust running loss & accuracy
        self.running_loss += loss
        accuracy = self.get_avg_batch_acc(predicted_logits, labels)
        self.running_acc += accuracy

        return

    def test(self, batch_size, dataloader, num_subsets, dataset_purpose):
        """
        Description:
            this function tests the model

        Parameters:
            batch_size: int, size of the batches of sentences processed at a time
            dataloader: torch dataloader
            num_subsets: int, number of subsets of the dataset
            dataset_purpose: str, "training"/"validation"/"test"

        Returns:
            avg_loss: float, loss computed between predicted outputs and ground truth values (for whole dataset)
            avg_accuracy: float, accuracy computed between predicted outputs and ground truth values (for whole dataset)
        """
        assert (
            dataset_purpose == "test" or dataset_purpose == "validation"
        ), '"dataset_purpose" is assigned an invalid string'

        # validation/test loss and accuracy calculated over a whole epoch
        self.running_loss, self.running_acc = 0.0, 0.0
        # for loop used because validation/test set is split into subsets
        self.num_batches = 0
        for i in range(num_subsets):
            self.run_subset(batch_size, dataloader, i, dataset_purpose=dataset_purpose)

        # average loss and accuracy calculated over the whole dataset
        avg_loss = self.running_loss / self.num_batches
        avg_accuracy = self.running_acc / self.num_batches

        return avg_loss, avg_accuracy

    def test_probe(
        self, batch_size, num_test_subsets=1, load_model=False, learning_rate=None
    ):
        test_dataloader = self.early_get_dataloader(
            num_test_subsets, "test", batch_size
        )

        if load_model is True:
            # initialize batch size and learning rate in probe trainer for proper loading of model parameters
            assert (
                learning_rate is not None
            ), "learning rate must have a value for proper loading of model parameters"
            self.init_test_variables(batch_size, learning_rate)
            self.load_model()  # load probe parameters

        _, avg_test_accuracy = self.test(
            batch_size,
            test_dataloader,
            num_test_subsets,
            dataset_purpose="test",
        )
        print(f"test set accuracy: {avg_test_accuracy:.2f}%")
        return avg_test_accuracy

    def init_test_variables(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # loss function
        self.loss_fn = self.get_loss_fn()

    def save_model(self, save_dir=None):
        if save_dir is None:
            save_dir = self.dataset_dir
        else:
            None

        os.makedirs(save_dir, exist_ok=True)  # create directory if it doesn't exist

        # create model file name
        batch_size = str(self.batch_size)
        learning_rate = str(self.learning_rate).replace(".", "p")
        model_file_name = f"probe_{batch_size}_{learning_rate}.pt"

        model_file_path = os.path.join(save_dir, model_file_name)
        torch.save(self.model.state_dict(), model_file_path)

        return

    def load_model(self, load_dir=None):
        if load_dir is None:
            load_dir = self.dataset_dir
        else:
            None

        # create model file name
        batch_size = str(self.batch_size)
        learning_rate = str(self.learning_rate).replace(".", "p")
        model_file_name = f"probe_{batch_size}_{learning_rate}.pt"

        model_file_path = os.path.join(load_dir, model_file_name)

        # load model parameters
        self.model.load_state_dict(
            torch.load(model_file_path, map_location=torch.device(self.device))
        )

        return


class probing_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        device,
        dataset_purpose: str,
        subset_num=0,
        dataset_dir=None,
    ):
        """
        Description:
            dataset that is used for running the probe

        Parameters:
            device: device which is used for computation (CPU/GPU/TPU)
            dataset_name: str, name of the investigated dataset
            dataset_purpose: str, "training"/"validation"/"test"
            subset_num: int, number of the subset for the dataset
        """

        self.device = device
        self.dataset_purpose = dataset_purpose
        self.subset_num = subset_num
        if not dataset_dir.endswith(dataset_purpose):
            self.dataset_dir = os.path.join(
                dataset_dir,
                self.dataset_purpose,
            )
        else:
            self.dataset_dir = dataset_dir

        self.X, self.Y = self.load_dataset()

        self.X.to(self.device)
        self.Y.to(self.device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_input_data(self):
        return self.X

    def update_input_data(self, updated_input):
        self.X = updated_input

    def load_dataset(self):
        input_file_name = f"input_tensor_{self.subset_num}.pt"
        output_file_name = f"output_tensor_{self.subset_num}.pt"

        input_data = torch.load(
            os.path.join(self.dataset_dir, input_file_name),
            map_location=torch.device(self.device),
        )
        output_data = torch.load(
            os.path.join(self.dataset_dir, output_file_name),
            map_location=torch.device(self.device),
        )

        return input_data, output_data


class data_truncator:
    def __init__(self, truncation_method, original_feature_size, updated_feature_size):
        self.truncation_method = truncation_method
        self.original_feature_size = original_feature_size
        self.updated_feature_size = updated_feature_size
        self.PCA_eigen_vectors = None

    def truncate_data(self, original_input, dataset_purpose):
        """
        Description:
            truncate the feature size of the input examples to fit probe dimensions

        Parameters:
            original_input : torch tensor, tensor containing the probing original input data
            dataset_purpose: str, "training"/"validation"/"test". Used only for PCA

        Returns:
            updated_input: torch tensor, truncated input data tensor
        """
        if (
            "average" in self.truncation_method.lower()
            or "avg" in self.truncation_method.lower()
        ):
            updated_data = self.truncate_avg(original_input)
        elif "pca" in self.truncation_method.lower():
            updated_data = self.truncate_PCA(original_input, dataset_purpose)

        return updated_data

    def truncate_avg(self, original_input):
        assert (
            self.original_feature_size % self.updated_feature_size == 0
        ), "model input dimensions must be a factor of the original feature size"
        scaling_factor = int(self.original_feature_size / self.updated_feature_size)
        original_input = original_input.view(original_input.size(0), -1, scaling_factor)
        updated_input = original_input.mean(dim=-1)

        return updated_input

    def truncate_PCA(self, original_input, dataset_purpose):
        # Step 1: Standardize the data
        mean = torch.mean(original_input, dim=0)
        standardized_data = original_input - mean

        if (
            "train" in dataset_purpose
        ):  # if training data, compute the eigen vectors for covariance matrix
            # Step 2: Compute the covariance matrix
            cov_matrix = torch.matmul(standardized_data.t(), standardized_data) / (
                standardized_data.size(0) - 1
            )
            # Step 3: Compute eigen values and eigen vectors
            eigen_values, eigen_vectors = torch.linalg.eig(cov_matrix)
            # Take the real part of the eigen values & eigen vectors
            eigen_values = eigen_values.real
            eigen_vectors = eigen_vectors.real
            # Step 4: Sort eigen values and corresponding eigen vectors in descending order
            sorted_indices = torch.argsort(eigen_values, descending=True)
            sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
            self.PCA_eigen_vectors = sorted_eigen_vectors

        if self.PCA_eigen_vectors is None:
            raise Exception(
                "PCA cannot be applied since there are no computed eigen vectors"
            )
        else:
            # Transform the data using the sorted eigen vectors (principal components)
            updated_input = torch.matmul(
                standardized_data,
                self.PCA_eigen_vectors[:, : self.updated_feature_size],
            )
            return updated_input
