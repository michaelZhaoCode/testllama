def extract_first_200_lines(input_filename, output_filename):
    try:

        with open(input_filename, 'r') as input_file:
            _ = [next(input_file) for _ in range(4000)]
            lines = [next(input_file) for _ in range(200)]

        with open(output_filename, 'w') as output_file:
            output_file.writelines(lines)

    except FileNotFoundError:
        print("The specified input file does not exist.")
    except StopIteration:
        print("The input file has less than 200 lines.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
extract_first_200_lines("books_large_p1.txt", "books_small.txt")
