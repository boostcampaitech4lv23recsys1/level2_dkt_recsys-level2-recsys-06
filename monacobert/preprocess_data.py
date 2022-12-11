# This code is based on the following repositories:
#  https://github.com/UpstageAI/cl4kt

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import os
import pickle

# Please specify your dataset Path
BASE_PATH = "/opt/ml/input/data"

# Concat Train & Test data
train_df = pd.read_csv(os.path.join(BASE_PATH,"train_data.csv"), sep=',')
test_df = pd.read_csv(os.path.join(BASE_PATH,"test_data.csv"), sep=',')
concat_df = pd.concat([train_df,test_df])
concat_df.to_csv(os.path.join(BASE_PATH,'concat.csv'), index=False)

min_user_inter_num = 1
data_path = os.path.join(BASE_PATH, "concat.csv")

df = pd.read_csv(data_path)
df = df.rename(
    columns={
        "userID": "user_id",
        "assessmentItemID": "item_id",
        "KnowledgeTag": "skill_id",
        "Timestamp": "timestamp",
        "answerCode": "correct"
    }
)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df["timestamp"] = df["timestamp"] - df["timestamp"].min()
df["timestamp"] = df["timestamp"].dt.total_seconds().astype(int)
df["skill_name"] = np.zeros(len(df), dtype=np.int64)

# Remove continuous outcomes
# df = df[df["correct"].isin([0, 1])]
df["correct"] = df["correct"].astype(np.int32)

# Filter too short sequences
df = df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
df["skill_id"] = np.unique(df["skill_id"].astype(str), return_inverse=True)[1]
with open(os.path.join(BASE_PATH, "skill_id_name"), "wb") as f:
    pickle.dump(dict(zip(df["skill_id"], df["skill_name"])), f)

# Build Q-matrix
Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
for item_id, skill_id in df[["item_id", "skill_id"]].values:
    Q_mat[item_id, skill_id] = 1

# Remove row duplicates due to multiple skills for one item
df = df.drop_duplicates(["user_id", "timestamp"])

print("# Users: {}".format(df["user_id"].nunique()))
print("# Skills: {}".format(df["skill_id"].nunique()))
print("# Items: {}".format(df["item_id"].nunique()))
print("# Interactions: {}".format(len(df)))

# # Get unique skill id from combination of all skill ids
# unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
# df["skill_id"] = unique_skill_ids[df["item_id"]]

# print("# Preprocessed Skills: {}".format(df["skill_id"].nunique()))
# # Sort data temporally
# df.sort_values(by="timestamp", inplace=True)

# # Sort data by users, preserving temporal order for each user
# df = pd.concat([u_df for _, u_df in df.groupby("user_id")])
# df.to_csv(os.path.join(BASE_PATH, "original_df.csv"), sep=",", index=False)

df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
df.reset_index(inplace=True, drop=True)

# Save data
with open(os.path.join(BASE_PATH, "question_skill_rel.pkl"), "wb") as f:
    pickle.dump(csr_matrix(Q_mat), f)

sparse.save_npz(os.path.join(BASE_PATH, "q_mat.npz"), csr_matrix(Q_mat))

df.to_csv(os.path.join(BASE_PATH, "preprocessed_df.csv"), sep=",", index=False)