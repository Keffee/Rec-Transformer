#Renew the user_features and video_features using the new mapped id
# each row for one user/video
import os
import pandas as pd

feature_dir = "/home/kfwang/20250613Rec-Factory/data/KuaiRand-27K/data"
tmp_dir = "/home/kfwang/20250613Rec-Factory/data/KuaiRand-27K/KuaiRand-27K-Processed"
#os.makedirs(tmp_dir, exist_ok=True)
# load the mappings.csv
user_id_mapping = pd.read_csv(os.path.join(tmp_dir, "mappings", "user_id_mapping.csv"))
item_id_mapping = pd.read_csv(os.path.join(tmp_dir, "mappings", "item_id_mapping.csv"))

user_features = pd.read_csv(os.path.join(feature_dir,"user_features_27k.csv"))  
user_id_mapping.columns = ["user_id", "continuous_user_id"]
user_features_mapped = user_features.merge(user_id_mapping, on="user_id", how="inner")
user_features_mapped.drop(columns=["user_id"], inplace=True)
user_features_mapped.rename(columns={"continuous_user_id": "user_id"}, inplace=True)
# Move 'user_id' to the first column
cols = user_features_mapped.columns.tolist()
cols = [cols[-1]] + cols[:-1]
user_features_mapped = user_features_mapped[cols]

user_features_mapped.to_csv(os.path.join(tmp_dir,"mapped_user_features_27k.csv"), index=False)

# video
# List of input files
video_feature_files = [
    "video_features_basic_27k.csv",
    "video_features_statistic_27k_part1.csv",
    "video_features_statistic_27k_part2.csv",
    "video_features_statistic_27k_part3.csv"
]
item_id_mapping.columns = ["video_id", "continuous_video_id"]
# Process each file
for file_name in video_feature_files:
    input_path = os.path.join(feature_dir, file_name)
    output_path = os.path.join(tmp_dir, f"mapped_{file_name}")

    print(f"Processing {file_name}...")

    df = pd.read_csv(input_path)

    # Make sure ID column is named correctly
    if "video_id" not in df.columns:
        raise ValueError(f"{file_name} is missing 'video_id' column")

    # Merge with mapping to get new item_id
    df_mapped = df.merge(item_id_mapping, on="video_id", how="inner")

    # Drop old ID and rename
    df_mapped.drop(columns=["video_id"], inplace=True)
    df_mapped.rename(columns={"continuous_video_id": "video_id"}, inplace=True)

    # Move 'video_id' to the first column
    cols = df_mapped.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    df_mapped = df_mapped[cols]

    # Save
    df_mapped.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")