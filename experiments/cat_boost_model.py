import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.pipeline import CVResult, load_train_test, save_results

catboost_params = {
    'iterations': 5000,
    'learning_rate': 0.04,
    'depth': 6,
    'l2_leaf_reg': 5.0,
    'eval_metric': 'AUC',
    'early_stopping_rounds': 300,
    'random_seed': 42,
    'verbose': 0,
}


def train_evaluate_cv(
    df_x: pd.DataFrame, df_y: pd.Series, cat_cols: list[str]
) -> CVResult:
    # Same folds as in xgboost_model.py, so the scores are directly comparable
    skf = StratifiedKFold(shuffle=True, random_state=42)
    y_pred = np.zeros(len(df_x))
    fold_scores: list[float] = []
    fold_models: list[CatBoostClassifier] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_x, df_y)):
        # CatBoost encodes categorical features itself, so no external encoder is needed
        train_pool = Pool(df_x.iloc[train_idx], df_y.iloc[train_idx], cat_features=cat_cols)
        val_pool = Pool(df_x.iloc[val_idx], df_y.iloc[val_idx], cat_features=cat_cols)

        model = CatBoostClassifier(**catboost_params)
        model.fit(train_pool, eval_set=val_pool)

        val_preds = model.predict_proba(val_pool)[:, 1]
        y_pred[val_idx] = val_preds
        fold_models.append(model)

        score = roc_auc_score(df_y.iloc[val_idx], val_preds)
        fold_scores.append(score)
        print(f'Fold {fold + 1}: AUC = {score:.5f} (best step: {model.get_best_iteration()})')

    return CVResult(y_pred, fold_scores, fold_models)


def predict_folds(
    fold_models: list[CatBoostClassifier], X_test: pd.DataFrame, cat_cols: list[str]
) -> np.ndarray:
    """Average the predictions of all fold models."""
    test_pool = Pool(X_test, cat_features=cat_cols)
    preds = np.zeros(len(X_test))
    for model in fold_models:
        preds += model.predict_proba(test_pool)[:, 1]
    return preds / len(fold_models)


def main():
    X_train, y_train, X_test, cat_cols = load_train_test()
    cv = train_evaluate_cv(X_train, y_train, cat_cols)
    test_preds = predict_folds(cv.fold_models, X_test, cat_cols)
    save_results('catboost', y_train, cv, X_test, test_preds)


if __name__ == '__main__':
    main()
