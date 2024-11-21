def get_vocab_size(file_path: str):
    """
    Calculates the vocabulary size from a text file.

    This function reads the text file specified by `file_path`, extracts the unique characters
    in the text, and returns the number of unique characters as the vocabulary size.

    Args:
        file_path (str): The path to the text file.

    Returns:
        int: The number of unique characters (vocabulary size) in the text.
    """
    # Open the file and read its content as a string
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract unique characters from the text and sort them
    chars = sorted(list(set(text)))

    # The vocabulary size is the number of unique characters
    vocab_size = len(chars)

    return vocab_size
