from sklearn.metrics import roc_auc_score

from src.data_utils import download_dataset, train_test_split_dataset
from pathlib import Path
from catboost import CatBoostClassifier, Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

df = download_dataset(str(DATA_DIR))
X_train, X_val, y_train, y_val, df_x = train_test_split_dataset(df)
cat_features = [col for col in df_x.columns]

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.04,
    depth=6,
    l2_leaf_reg=5.0,
    eval_metric='AUC',
    early_stopping_rounds=100,
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=val_pool)
y_proba = model.predict_proba(val_pool)[:, 1]
print(roc_auc_score(y_val, y_proba))
