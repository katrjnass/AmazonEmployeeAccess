"""Blend XGBoost and CatBoost.

Run xgboost_model.py and cat_boost_model.py first: they save the OOF predictions
(models/*_oof.npy) and the test predictions (submission_*.csv) used here.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.data_utils import DATA_DIR, MODELS_DIR, PROJECT_ROOT

MODELS = ['xgb', 'catboost']
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)  # weight of CatBoost, XGBoost gets 1 - w


def to_rank(preds: np.ndarray) -> np.ndarray:
    """
    Probabilities -> ranks in [0, 1]. AUC only depends on order,
    so the models' scales stop mattering.
    """
    return (rankdata(preds) - 1) / (len(preds) - 1)


def blend(xgb_preds: np.ndarray, cb_preds: np.ndarray, w: float, use_rank: bool) -> np.ndarray:
    if use_rank:
        xgb_preds, cb_preds = to_rank(xgb_preds), to_rank(cb_preds)
    return (1 - w) * xgb_preds + w * cb_preds


def search_weights(y: np.ndarray, xgb_oof: np.ndarray, cb_oof: np.ndarray) -> pd.DataFrame:
    rows = []
    for use_rank in (False, True):
        for w in WEIGHTS:
            auc = roc_auc_score(y, blend(xgb_oof, cb_oof, w, use_rank))
            rows.append(
                {'method': 'rank' if use_rank else 'proba', 'catboost_weight': w, 'oof_auc': auc}
            )
    return pd.DataFrame(rows)


def main():
    y = pd.read_csv(DATA_DIR / 'train.csv')['ACTION'].values
    xgb_oof, cb_oof = (np.load(MODELS_DIR / f'{name}_oof.npy') for name in MODELS)
    xgb_test, cb_test = (pd.read_csv(PROJECT_ROOT / f'submission_{name}.csv') for name in MODELS)
    assert (xgb_test['id'].values == cb_test['id'].values).all(), 'test ids do not match'

    print(f'XGBoost  OOF AUC: {roc_auc_score(y, xgb_oof):.5f}')
    print(f'CatBoost OOF AUC: {roc_auc_score(y, cb_oof):.5f}')
    print(f'Correlation of OOF predictions: {np.corrcoef(xgb_oof, cb_oof)[0, 1]:.4f}')

    results = search_weights(y, xgb_oof, cb_oof)
    for method, group in results.groupby('method'):
        best = group.loc[group['oof_auc'].idxmax()]
        equal = group.loc[group['catboost_weight'] == 0.5, 'oof_auc'].item()
        print(f'{method:5s}: 50/50 = {equal:.5f}, best = {best.oof_auc:.5f} '
              f'(CatBoost weight {best.catboost_weight:.2f})')

    best = results.loc[results['oof_auc'].idxmax()]
    use_rank = best['method'] == 'rank'
    test_preds = blend(xgb_test['ACTION'].values, cb_test['ACTION'].values,
                       best['catboost_weight'], use_rank)

    submission = pd.DataFrame({'id': xgb_test['id'], 'ACTION': test_preds})
    submission_path = PROJECT_ROOT / 'submission_blend.csv'
    submission.to_csv(submission_path, index=False)
    print(f'Saved submission ({best.method}, CatBoost weight {best.catboost_weight:.2f}) '
          f'to {submission_path}')


if __name__ == '__main__':
    main()
