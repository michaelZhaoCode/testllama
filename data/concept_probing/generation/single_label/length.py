import csv
import os


def average_line_length(filename):
    try:
        with open(filename) as file:
            reader = csv.reader(file)
            total_lines = 0
            total_length = 0
            for line in reader:
                # Join the line to treat it as a single string, then measure its length
                line_string = ",".join(line)
                total_length += len(line_string)
                total_lines += 1
            if total_lines > 0:
                # Calculate the average length
                average_length = total_length / total_lines
                return average_length
            else:
                return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_all_csvs(directory="."):
    # List all files in the given directory
    for filename in os.listdir(directory):
        # Process only CSV files
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            avg_length = average_line_length(file_path)
            if avg_length is not None:
                print(
                    f"The average line length in {filename} is {avg_length:.2f} characters."
                )


# Example usage
print("Ambition lengths")
process_all_csvs("./ambition/michael/csv")
print("Not Ambition lengths")
process_all_csvs("./not_ambition/michael/csv")
