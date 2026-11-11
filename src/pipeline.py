"""Steps shared by all model scripts: loading the data and saving the CV results."""
from dataclasses import dataclass
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_utils import DATA_DIR, MODELS_DIR, PROJECT_ROOT, download_dataset, create_test_df
from src.features import build_train_test


@dataclass
class CVResult:
    """Output of a model's train_evaluate_cv."""

    oof_preds: np.ndarray       # out-of-fold prediction for every train row
    fold_scores: list[float]    # ROC-AUC of each fold
    fold_models: list[Any]      # one fitted model (or (encoder, model) pair) per fold


def load_train_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Load train and test and build features. X_test is indexed by id."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = download_dataset(str(DATA_DIR))
    test = create_test_df(str(DATA_DIR))
    return build_train_test(train, test)


def save_results(
    name: str, y_train: pd.Series, cv: CVResult, X_test: pd.DataFrame, test_preds: np.ndarray
) -> None:
    """Print CV scores, save OOF predictions, fold models and submission_<name>.csv."""
    print(f'Mean fold score: {np.mean(cv.fold_scores):.5f} ± {np.std(cv.fold_scores):.5f}')
    print(f'Final OOF ROC-AUC: {roc_auc_score(y_train, cv.oof_preds):.5f}')

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / f'{name}_oof.npy', cv.oof_preds)  # OOF predictions for blending

    models_path = MODELS_DIR / f'{name}_folds.joblib'
    joblib.dump(cv.fold_models, models_path)
    print(f'Saved {len(cv.fold_models)} fold models to {models_path}')

    submission = pd.DataFrame({'id': X_test.index, 'ACTION': test_preds})
    submission_path = PROJECT_ROOT / f'submission_{name}.csv'
    submission.to_csv(submission_path, index=False)
    print(f'Saved submission to {submission_path}')
