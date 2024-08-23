import gc

import pandas as pd
import torch


def load_csv_to_list(csv_path):
    df = pd.read_csv(csv_path, header=None)
    list = df.values.tolist()

    return list


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


def save_torch_tensor(input_tensor, tensor_path):
    """
    Description:
        save a torch tensor in the

    Parameters:
        input_tensor: torch tensor, tensor that is intended to be saved
        tensor_path: string, file path for the saved tensor

    Returns:
        None
    """

    # saving the tensor
    torch.save(input_tensor, tensor_path)

    return


def memory_usage(message: str = ""):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / (2**20)  # Convert bytes to MB
    reserved = torch.cuda.memory_reserved() / (2**20)  # Convert bytes to MB
    maximum = torch.cuda.max_memory_allocated() / (2**20)  # Convert bytes to MB
    print(
        f"{message}Memory - Allocated: {allocated:.2f} KB, Reserved: {reserved:.2f} KB, Peak: {maximum:.2f} KB"
    )

    return [allocated, reserved, maximum]


def get_avg_num_words(input_list):
    total_num_words = 0
    num_list_items = len(input_list)
    for list_item in input_list:
        num_words = len(list_item.split(" "))
        total_num_words += num_words

    return total_num_words / num_list_items


def get_avg_batch_acc(predicted_logits, ground_truth):
    """
    Description:
        this function takes the linear output of the NN and applies argmax function, then it computes the accuracy

    Parameters:
        predicted_logits: torch tensor, predicted logits from the model
        ground_truth: torch tensor, dataset labels

    Returns:
        avg_batch_acc: float, batch accuracy computed between the predicted and ground truth values
    """

    batch_size = ground_truth.shape[0]

    predicted_classes = torch.argmax(predicted_logits, dim=-1)

    # Element-wise comparison
    comparison = predicted_classes == ground_truth

    # Count the number of equal elements
    num_equal_elements = comparison.sum().item()

    avg_batch_acc = num_equal_elements / batch_size

    return avg_batch_acc


def arithmatic_series(begin_num, end_num, num_elements):
    return (begin_num + end_num) * num_elements / 2
