import os
import math
import pandas as pd
from typing import Optional
from multiprocessing import Pool
from tqdm import tqdm


def to_seq_data(
    ratings_data: pd.DataFrame,
    user_data: Optional[pd.DataFrame] = None,
    seq_prefix: str = "sequence_"
) -> pd.DataFrame:
    # Join user data if provided
    if user_data is not None:
        ratings_data_transformed = ratings_data.join(
            user_data.set_index("user_id"), on="user_id"
        )
    else:
        ratings_data_transformed = ratings_data

    # Automatically find list-like columns to stringify (skip user_id)
    seq_columns = [col for col in ratings_data_transformed.columns if col != "user_id"]

    for col in seq_columns:
        ratings_data_transformed[col] = ratings_data_transformed[col].apply(
            lambda x: ",".join(str(v) for v in x) if isinstance(x, (list, tuple)) else str(x)
        )

    # Rename columns to add a prefix for clarity
    rename_map = {col: f"{seq_prefix}{col}" for col in seq_columns}
    ratings_data_transformed.rename(columns=rename_map, inplace=True)
    return ratings_data_transformed


def _save_chunk(args):
    chunk_df, file_path = args
    chunk_df.to_csv(file_path, index=False, sep=",")


def save_multifile_csv_parallel(df: pd.DataFrame, save_dir: str, file_prefix: str, num_files: int):
    os.makedirs(save_dir, exist_ok=True)
    chunk_size = math.ceil(len(df) / num_files)
    tasks = []

    # Prepare tasks (DataFrame chunks + file names)
    for i in range(num_files):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start:end]
        file_path = os.path.join(save_dir, f"{file_prefix}_{i}.csv")
        tasks.append((chunk, file_path))

    # Parallel saving with progress bar
    with Pool(processes=num_files) as pool:
        list(tqdm(pool.imap_unordered(_save_chunk, tasks), total=num_files, desc="Saving chunks"))

    # Save users.csv (row counts per chunk)
    users_file = os.path.join(save_dir, f"{file_prefix}_users.csv")
    with open(users_file, "w") as f:
        for i, (chunk, _) in enumerate(tasks):
            count = chunk.shape[0]
            f.write(f"{i},{count}\n")
    print(f"Saved {num_files} chunks + users file to {save_dir}")


# Load and concat data
df1_part1 = pd.read_csv("/home/kfwang/20250613Rec-Factory/data/KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part1.csv")
df1_part2 = pd.read_csv("/home/kfwang/20250613Rec-Factory/data/KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part2.csv")
df2_part1 = pd.read_csv("/home/kfwang/20250613Rec-Factory/data/KuaiRand-27K/data/log_standard_4_22_to_5_08_27k_part1.csv")
df2_part2 = pd.read_csv("/home/kfwang/20250613Rec-Factory/data/KuaiRand-27K/data/log_standard_4_22_to_5_08_27k_part2.csv")

df_all = pd.concat([df1_part1, df1_part2, df2_part1, df2_part2], ignore_index=True)
print(f"records before filter: {df_all.shape}")


# Filter users/items with at least 5 interactions
item_id_count = df_all["video_id"].value_counts().rename_axis("unique_values").reset_index(name="item_count")
print(item_id_count.shape)
print(item_id_count)
user_id_count = df_all["user_id"].value_counts().rename_axis("unique_values").reset_index(name="user_count")
print(user_id_count.shape)
print(user_id_count)

df_all = df_all.join(item_id_count.set_index("unique_values"), on="video_id")
df_all = df_all.join(user_id_count.set_index("unique_values"), on="user_id")
df_all = df_all[(df_all["item_count"] >= 5) & (df_all["user_count"] >= 5)]
print(f"#records after filter: {df_all.shape}")


# Map to contiguous IDs
cat_item = pd.Categorical(df_all["video_id"])
cat_user = pd.Categorical(df_all["user_id"])
df_all["video_id"] = cat_item.codes
df_all["user_id"] = cat_user.codes
item_id_mapping = pd.DataFrame({
    "original_item_id": cat_item.categories,
    "continuous_item_id": range(len(cat_item.categories))
})
user_id_mapping = pd.DataFrame({
    "original_user_id": cat_user.categories,
    "continuous_user_id": range(len(cat_user.categories))
})
print(f"#users after filter: {df_all['user_id'].nunique()}")
print(f"#items after filter: {df_all['video_id'].nunique()}")
num_unique_items = df_all["video_id"].nunique()
df_all.rename(columns={'video_id': 'item_ids'}, inplace=True)
df_all.rename(columns={'time_ms': 'timestamps'}, inplace=True)


# Group by user and meanwhile keep users with sequence length larger than 5 
df_all_filtered = (
    df_all
    .sort_values(by=["user_id", "timestamps"])
    .groupby("user_id")
    .filter(lambda x: len(x) >= 5)
)


# Add flag for positive interactions
interaction_cols = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'is_profile_enter']
df_all_filtered['is_recall_candidate'] = (df_all_filtered['is_hate'] == 0) & (df_all_filtered[interaction_cols].sum(axis=1) >= 1)


# Group into sequences
df_all_group = df_all_filtered.groupby("user_id")
group_cols = ['item_ids', 'timestamps', 'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate', 'long_view', 'play_time_ms', 'duration_ms', 'profile_stay_time', 'comment_stay_time', 'is_profile_enter', 'tab', 'is_recall_candidate']
seq_data = pd.DataFrame({
    "user_id": list(df_all_group.groups.keys()),
    **{col: list(df_all_group[col].apply(list)) for col in group_cols}
})
print(f"sequence length stats:")
print("Mean:", seq_data["item_ids"].apply(len).mean())
print("Min:", seq_data["item_ids"].apply(len).min())
print("Max:", seq_data["item_ids"].apply(len).max())


# Save data into files
save_dir = "/home/kfwang/20250613Rec-Factory/data/KuaiRand-27K/KuaiRand-27K-Processed"
seq_data = to_seq_data(seq_data).reset_index(drop=True)
save_multifile_csv_parallel(seq_data, save_dir, "sasrec_format", num_files=4)


# Save mappings
mapping_dir = os.path.join(save_dir, "mappings")
os.makedirs(mapping_dir, exist_ok=True)
item_id_mapping.to_csv(os.path.join(mapping_dir, "item_id_mapping.csv"), index=False)
user_id_mapping.to_csv(os.path.join(mapping_dir, "user_id_mapping.csv"), index=False)
print(f"Saved ID mappings to {mapping_dir}")

# train/test split timestamp: split_ts = 1651795200000, train_ts<=split_ts