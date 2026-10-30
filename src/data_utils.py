import kagglehub
import os
import pandas as pd

from sklearn.model_selection import train_test_split


def download_dataset(path):
    if not os.path.exists(path):
        kagglehub.competition_download('amazon-employee-access-challenge', output_dir=path)
    else:
        print('The data is already on disk, so we skip the download')

    df = pd.read_csv(os.path.join(path, 'train.csv'))
    return df


def train_test_split_dataset(df):
    df_y = df['ACTION']
    df_x = df.drop(columns=['ACTION', 'ROLE_CODE'])
    X_train, X_val, y_train, y_val = train_test_split(df_x, df_y, test_size=0.2, random_state=42, stratify=df_y)
    return X_train, X_val, y_train, y_val, df_x