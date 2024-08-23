import os
import re

import pandas as pd


INVESTIGATED_CONCEPT = "ambition"


def check_any_n_equal(row, cols, n):
    # Select the specific columns
    selected_values = row[cols]
    # Create a set of unique elements in the selected columns
    unique_elements = set(selected_values)
    # Check if there is any element that appears at least n times
    for element in unique_elements:
        if list(selected_values).count(element) >= n:
            return True
    return False


def count_borderline(row, cols):
    return sum(row[cols] == "TRUE")


def get_mode(row, cols):
    return row[cols].mode()[0]


def determine_valid_label(row, human_annotation_cols, human_borderline_cols):
    """
    there are 3 conditions for which the labels can be deemed acceptable:
    1. there is a perfect agreement on the label with less than 2 borderline declarations
    2. there is a semi-perfect agreement on the label (3 vs 1) with no borderline declarations
    3. there is a semi-perfect agreement on the label (3 vs 1) with only 1 borderline declaration, and that being
        for the annotator which gave the different label than the other 3

    these conditions are used such that changing the labels based on borderline would've still led to having a
    semi-perfect agreement (3 vs 1)
    """
    if (row["perfect_agreement"] is True and row["num_borderline"] <= 1) or (
        row["semi_perfect_agreement"] is True and row["num_borderline"] == 0
    ):
        return True
    elif row["semi_perfect_agreement"] is True and row["num_borderline"] == 1:
        diff_col_L = [col for col in human_annotation_cols if row[col] != row["label"]]
        diff_col_integer = int(re.search(r"\d+", diff_col_L[0]).group())
        for col in human_borderline_cols:
            if str(diff_col_integer) in col and row[col] == "TRUE":
                return True

    return False


if __name__ == "__main__":
    human_labelled_csv_path = os.path.join(
        os.getcwd(), "data", "concept_probing", "generation", "human_labelled"
    )
    concept_file_name = f"{INVESTIGATED_CONCEPT}.csv"

    concept_df = pd.read_csv(os.path.join(human_labelled_csv_path, concept_file_name))
    # remove first row of strings
    concept_df.drop(index=0, inplace=True)
    concept_df.reset_index(drop=True, inplace=True)

    annotators = ["michael", "mohamed", "michelle", "sihan"]
    unnamed_columns = [col for col in concept_df.columns if col.startswith("Unnamed")]

    for i, element in enumerate(annotators):
        concept_df.rename(columns={element: f"annotator_{i}_L"}, inplace=True)
        concept_df[f"annotator_{i}_L"] = concept_df[f"annotator_{i}_L"].astype(
            int
        )  # change labels datatype to int
    for i, element in enumerate(unnamed_columns):
        concept_df.rename(columns={element: f"annotator_{i}_B"}, inplace=True)

    # change gpt4's labels' datatype to int
    concept_df["gpt4"] = concept_df["gpt4"].astype(int)

    # concept_df = concept_df.head(300)

    # Define a regular expression pattern to get human annotation columns
    pattern = r"^annotator_\d+_L$"
    human_annotation_cols = [
        col for col in concept_df.columns if re.match(pattern, col)
    ]

    # Apply perfect agreement function
    concept_df["perfect_agreement"] = concept_df.apply(
        check_any_n_equal, axis=1, cols=human_annotation_cols, n=4
    )
    # Apply semi-perfect agreement function
    concept_df["semi_perfect_agreement"] = concept_df.apply(
        check_any_n_equal, axis=1, cols=human_annotation_cols, n=3
    )

    concept_df["label"] = concept_df.apply(get_mode, axis=1, cols=human_annotation_cols)
    try:
        # Convert column 'label' from string to integer
        concept_df["label"] = concept_df["label"].astype(int)
    except Exception:
        pass

    # Define a regular expression pattern to get human borderline columns
    pattern = r"^annotator_\d+_B$"
    human_borderline_cols = [
        col for col in concept_df.columns if re.match(pattern, col)
    ]

    # Count the number of borderline values in the specified columns for each row
    concept_df["num_borderline"] = concept_df.apply(
        count_borderline, axis=1, cols=human_borderline_cols
    )

    concept_df["valid_label"] = concept_df.apply(
        determine_valid_label,
        axis=1,
        human_annotation_cols=human_annotation_cols,
        human_borderline_cols=human_borderline_cols,
    )

    columns_interest = ["examples", "label"]
    # this conversion is for the code to pass pre-commit check
    concept_df["valid_label"] = concept_df["valid_label"].astype(int)
    validated_df = concept_df.drop(concept_df[concept_df["valid_label"] == 0].index)[
        columns_interest
    ]
    # validated_df = concept_df.drop(
    #     concept_df[concept_df["valid_label"] == False].index
    # )[columns_interest]
    validated_df.reset_index(drop=True, inplace=True)

    agreement_ratio = len(validated_df) / len(concept_df)

    print(f"annotator agreement: {agreement_ratio*100:.2f}%")

    # get labelling accuracy
    concept_df["human_vs_gpt4_label_match"] = concept_df["label"] == concept_df["gpt4"]
    # Calculate the accuracy
    accuracy = concept_df["human_vs_gpt4_label_match"].mean()
    # Print the accuracy
    print(f"human vs gpt4 labelling accuracy: {accuracy * 100:.2f}%")

    # Save DataFrames to a CSV file
    concept_df_file_name = f"{INVESTIGATED_CONCEPT}_unvalidated.csv"
    concept_df[columns_interest].to_csv(
        os.path.join(human_labelled_csv_path, concept_df_file_name), index=False
    )

    valid_df_file_name = f"{INVESTIGATED_CONCEPT}_validated.csv"
    validated_df.to_csv(
        os.path.join(human_labelled_csv_path, valid_df_file_name), index=False
    )

    df_no_negative_file_name = f"{INVESTIGATED_CONCEPT}_no_neg.csv"
    df_no_negative = validated_df[validated_df["label"] != -1]
    df_no_negative.to_csv(
        os.path.join(human_labelled_csv_path, df_no_negative_file_name), index=False
    )
