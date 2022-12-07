import os
import pickle
import pandas as pd
from pathlib import Path

DTYPE = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}

def main():
    train_df = pd.read_csv("/opt/ml/input/train_data.csv", dtype = DTYPE, parse_dates = ['Timestamp'], low_memory = False)\
                    .sort_values(by = ['userID', 'Timestamp']).reset_index(drop=True)

    test_df = pd.read_csv("/opt/ml/input/test_data.csv", low_memory = False)\
                .sort_values(by = ['userID', 'Timestamp']).reset_index(drop = True)
    
    total_df = pd.concat([train_df, test_df], axis = 0)
    total_df = total_df[total_df['answerCode'] != -1]
    grouped_dict = dict(total_df.groupby('assessmentItemID')['answerCode'].value_counts())

    dpath = Path(os.path.join("/opt/ml/output/asset/grouped_dict.pkl"))
    dpath.parent.mkdir(parents=True, exist_ok=True)
    with open(str(dpath), 'wb') as f:
        pickle.dump(grouped_dict, f)

if __name__ == '__main__':
    main()