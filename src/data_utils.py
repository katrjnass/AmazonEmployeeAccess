import os
from pathlib import Path
import kagglehub
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'


def download_dataset(path: str) -> pd.DataFrame:
    train_path = os.path.join(path, 'train.csv')
    if not os.path.exists(train_path):
        kagglehub.competition_download('amazon-employee-access-challenge', output_dir=path)
    else:
        print('The data is already on disk, so we skip the download')

    df = pd.read_csv(train_path)
    return df


def create_test_df(path: str) -> pd.DataFrame:
    test_path = os.path.join(path, 'test.csv')
    df_test = pd.read_csv(test_path)
    df_test = df_test.drop(columns=['ROLE_CODE', 'ROLE_FAMILY'])
    return df_test
