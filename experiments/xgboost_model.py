from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from category_encoders import CatBoostEncoder

from src.data_utils import download_dataset
from src.features import create_clean_pair, create_frequency_cols

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'learning_rate': 0.04,
    'max_depth': 6,
    'reg_lambda': 5.0,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'random_state': 42,
    'n_estimators': 2500,
    'early_stopping_rounds': 100,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df_y = df['ACTION']
    df_x = df.drop(columns=['ACTION', 'ROLE_CODE', 'ROLE_FAMILY'])

    df_x['ROLE_HIERARCHY'] = (
            df_x['ROLE_ROLLUP_1'].astype(str) + '_' + df_x['ROLE_ROLLUP_2'].astype(str)
    )
    df_x['MGR_ROLE_ROLLUP_2'] = (
            df_x['MGR_ID'].astype(str) + '_' + df_x['ROLE_ROLLUP_2'].astype(str)
    )
    df_x['MGR_ROLE_FAMILY_DESC'] = create_clean_pair(df_x['MGR_ID'], df_x['ROLE_FAMILY_DESC'])
    df_x['MGR_ROLE_TITLE'] = create_clean_pair(df_x['MGR_ID'], df_x['ROLE_TITLE'])
    df_x['RESOURCE_ROLE_TITLE'] = create_clean_pair(df_x['RESOURCE'], df_x['ROLE_TITLE'])
    df_x['RESOURCE_ROLE_DEPTNAME'] = create_clean_pair(df_x['RESOURCE'], df_x['ROLE_DEPTNAME'])
    df_x['RESOURCE_ROLE_ROLLUP_2'] = create_clean_pair(df_x['RESOURCE'], df_x['ROLE_ROLLUP_2'])
    df_x['RESOURCE_ROLE_FAMILY_DESC'] = create_clean_pair(
        df_x['RESOURCE'], df_x['ROLE_FAMILY_DESC']
    )

    freq_cols = create_frequency_cols(df_x, ['RESOURCE', 'MGR_ID', 'ROLE_DEPTNAME'])
    cat_cols = [col for col in df_x.columns if col not in freq_cols]

    return df_x, df_y, cat_cols


def train_evaluate_cv(
    df_x: pd.DataFrame, df_y: pd.Series, cat_cols: list[str]
) -> tuple[np.ndarray, list[float]]:
    skf = StratifiedKFold(shuffle=True, random_state=42)
    y_pred = np.zeros(len(df_x))
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_x, df_y)):
        y_train = df_y.iloc[train_idx]
        y_val = df_y.iloc[val_idx]

        encoder = CatBoostEncoder(cols=cat_cols, a=10.0, random_state=42)
        X_train_enc = encoder.fit_transform(df_x.iloc[train_idx], y_train)
        X_val_enc = encoder.transform(df_x.iloc[val_idx])

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train_enc,
            y_train,
            eval_set=[(X_val_enc, y_val)],
            verbose=False,
        )

        val_preds = model.predict_proba(X_val_enc)[:, 1]
        y_pred[val_idx] = val_preds

        score = roc_auc_score(y_val, val_preds)
        fold_scores.append(score)
        print(f'Fold {fold + 1}: AUC = {score:.5f} (best step: {model.best_iteration})')

    return y_pred, fold_scores


def main():
    df = download_dataset(str(DATA_DIR))
    df_x, df_y, cat_cols = prepare_features(df)
    y_pred, fold_scores = train_evaluate_cv(df_x, df_y, cat_cols)

    print(f'Mean fold score: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}')
    print(f'Final OOF ROC-AUC with encoding: {roc_auc_score(df_y, y_pred):.5f}')


if __name__ == '__main__':
    main()
