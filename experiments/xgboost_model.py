import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from category_encoders import CatBoostEncoder

from src.pipeline import CVResult, load_train_test, save_results

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'learning_rate': 0.01,
    'max_depth': 6,
    'reg_lambda': 5.0,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'random_state': 42,
    'n_estimators': 8000,
    'early_stopping_rounds': 300,
}

FoldModels = list[tuple[CatBoostEncoder, xgb.XGBClassifier]]


def train_evaluate_cv(
    df_x: pd.DataFrame, df_y: pd.Series, cat_cols: list[str]
) -> CVResult:
    skf = StratifiedKFold(shuffle=True, random_state=42)
    y_pred = np.zeros(len(df_x))
    fold_scores: list[float] = []
    fold_models: FoldModels = []

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
        fold_models.append((encoder, model))

        score = roc_auc_score(y_val, val_preds)
        fold_scores.append(score)
        print(f'Fold {fold + 1}: AUC = {score:.5f} (best step: {model.best_iteration})')

    return CVResult(y_pred, fold_scores, fold_models)


def predict_folds(fold_models: FoldModels, X_test: pd.DataFrame) -> np.ndarray:
    """Average the predictions of all fold models (each with its own encoder)."""
    preds = np.zeros(len(X_test))
    for encoder, model in fold_models:
        preds += model.predict_proba(encoder.transform(X_test))[:, 1]
    return preds / len(fold_models)


def main():
    X_train, y_train, X_test, cat_cols = load_train_test()
    cv = train_evaluate_cv(X_train, y_train, cat_cols)
    test_preds = predict_folds(cv.fold_models, X_test)
    save_results('xgb', y_train, cv, X_test, test_preds)


if __name__ == '__main__':
    main()
