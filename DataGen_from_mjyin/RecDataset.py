from datasets import Dataset, load_dataset
import pandas as pd
import logging

def remove_duplicate_rows(dataset: Dataset, subset: list = None, keep: str = 'first') -> pd.DataFrame:
    """
    Removes duplicate rows from a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame.
        subset (list, optional): A list of column names to consider when identifying duplicates.
                                 If None, all columns are used. Defaults to None.
        keep (str): Determines which duplicates to keep.
                    'first': Keep the first occurrence of a duplicate row.
                    'last': Keep the last occurrence of a duplicate row.
                    False: Drop all occurrences of duplicate rows (if a row appears more than once,
                           all instances will be removed).
                    Defaults to 'first'.

    Returns:
        pd.DataFrame: A new DataFrame with duplicate rows removed.
    """
    df = dataset.to_pandas()
    # Input validation: Ensure the input is a Pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    
    # Check and print the number of duplicate rows found (optional)
    # This step helps you understand how many rows will be removed before the operation
    num_duplicate_rows = df.duplicated(subset=subset).sum()
    if num_duplicate_rows > 0:
        logging.info(f"Found {num_duplicate_rows} duplicate rows before removal.")
    else:
        logging.info("No duplicate rows found.")

    # Remove duplicate rows using Pandas drop_duplicates method
    df_cleaned = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    logging.info(f"After removing duplicates, {len(df_cleaned)} records remain.")

    return Dataset.from_pandas(df_cleaned)

def k_core_filtering(dataset: Dataset, user_column: str = 'user_id', item_column: str = 'item_id', k_core: int = 5):
    # Convert the Hugging Face Dataset to a pandas DataFrame for easier manipulation.
    df = dataset.to_pandas()
    logging.info(f"Original dataset size: {len(df)} interaction records.")
    prev_len = -1
    current_len = len(df)

    iteration = 0
    while current_len != prev_len:
        iteration += 1
        prev_len = current_len
        user_counts = df.groupby(user_column).size()
        valid_users = user_counts[user_counts >= k_core].index
        df = df[df[user_column].isin(valid_users)]

        item_counts = df.groupby(item_column).size()
        valid_items = item_counts[item_counts >= k_core].index
        df = df[df[item_column].isin(valid_items)]

        current_len = len(df)
        logging.info(f"Iteration {iteration}: Dataset size reduced to {current_len} interaction records.")

    logging.info(f"{k_core}-core filtering completed. Final dataset size: {len(df)} interaction records.\n")
    return Dataset.from_pandas(df)

def serialize_user_interactions(dataset: Dataset, max_length: int, user_column: str = 'user_id',
                                item_column: str = 'item_id',
                                timestamp_column: str = 'timestamp',
                                sequence_column_name: str = 'item_sequence'):

    df = dataset.to_pandas()
    logging.info(f"Starting serialization for {len(df)} interaction records.")

    if pd.api.types.is_numeric_dtype(df[timestamp_column]):
        # Assume it's a Unix timestamp (seconds or milliseconds)
        # You might need to adjust 'unit' based on your timestamp's granularity (e.g., 's', 'ms', 'us', 'ns')
        # For this example, let's assume it's in seconds if it's a generic integer.
        logging.info(f"Assuming '{timestamp_column}' is a numeric (e.g., Unix) timestamp.")
    else:
        # Attempt to convert to datetime if it's a string or object type
        try:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            logging.info(f"Converted '{timestamp_column}' to datetime objects.")
        except Exception as e:
            logging.warning(f"Warning: Could not convert '{timestamp_column}' to datetime. "
                  f"Ensure it's a sortable format (e.g., Unix timestamp or ISO format). Error: {e}")
            logging.warning("Proceeding with existing timestamp column for sorting.")


    # Group by user_id and sort interactions by timestamp within each group.
    # Then, aggregate the item_id into a list.
    df_sorted = df.sort_values(by=[user_column, timestamp_column])
    def aggregate_sequences(group: pd.DataFrame) -> pd.Series:
        """
        Aggregates item_id and timestamp into separate lists for a given user group.
        """
        return pd.Series({
            sequence_column_name: group[item_column].tolist()[-max_length:],
            'timestamp_sequence': group[timestamp_column].tolist()[-max_length:],
        })

    # Apply the custom aggregation function
    user_sequences_df = df_sorted.groupby(user_column).apply(aggregate_sequences).reset_index()

    logging.info(f"Serialization complete. Created sequences for {len(user_sequences_df)} unique users.")

    # Convert the resulting pandas DataFrame back to a Hugging Face Dataset.
    return Dataset.from_pandas(user_sequences_df)

def split(sample):
    sample['item_sequence'] = sample['item_sequence'][:-1]
    sample['timestamp_sequence'] = sample['timestamp_sequence'][:-1]
    return sample

def create_dataset(file_path, max_seq_len, user_column='user_id', item_column='item_id', timestamp_column='timestamp', k_core=5, mode='train'):
    dataset = load_dataset('json', data_files=file_path, split='train')
    dataset = remove_duplicate_rows(dataset, subset=[user_column, item_column], keep='first')
    dataset = k_core_filtering(dataset, user_column=user_column, item_column=item_column, k_core=k_core)
    dataset = serialize_user_interactions(dataset, max_seq_len, user_column=user_column, item_column=item_column, timestamp_column=timestamp_column)
    # if mode == 'train':
    #     dataset = dataset.map(split, load_from_cache_file=True, num_proc=4)
    # dataset.to_parquet(file_path.replace('.jsonl', '.parquet'))  # Save the processed dataset
    return dataset

