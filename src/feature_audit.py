import itertools
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def calc_density_metrics(series: pd.Series, min_freq: int = 5) -> dict[str, float]:
    n_rows = len(series)
    counts = series.value_counts()
    n_unique = len(counts)

    singletons = (counts == 1).sum()
    rows_in_rare = counts[counts < min_freq].sum()

    return {
        'cardinality_ratio': n_unique / n_rows,
        'singleton_ratio': singletons / n_unique,
        'rare_rows_coverage': rows_in_rare / n_rows,
    }


def calc_target_metrics(series: pd.Series, y: pd.Series, min_freq: int = 5) -> dict[str, float]:
    counts = series.value_counts()
    valid_cats = counts[counts >= min_freq].index

    mask = series.isin(valid_cats)
    if mask.sum() > 0 and len(valid_cats) > 1:
        group_means = y[mask].groupby(series[mask]).mean()
        target_std = float(group_means.std())
    else:
        target_std = 0.0

    cat_codes = pd.factorize(series)[0].reshape(-1, 1)
    mi = float(mutual_info_classif(cat_codes, y, discrete_features=True, random_state=42)[0])
    return {'target_std': target_std, 'mutual_info': mi}


def audit_feature(
        series: pd.Series, y: pd.Series, min_freq: int = 5, as_dataframe: bool = False
) -> pd.DataFrame | dict[str, float]:
    metrics = {}
    metrics.update(calc_density_metrics(series, min_freq=min_freq))
    metrics.update(calc_target_metrics(series, y, min_freq=min_freq))

    if as_dataframe:
        return pd.DataFrame([metrics])
    return metrics


def screen_feature_pairs(
        df: pd.DataFrame, y: pd.Series, cat_cols: list[str], top_n: int = 10, min_freq: int = 5
) -> pd.DataFrame:
    results = []
    pairs = list(itertools.combinations(cat_cols, 2))

    for col1, col2 in pairs:
        pair_name = f'{col1}__{col2}'
        clean_pair = df[col1].astype(str) + '_' + df[col2].astype(str)
        metrics = audit_feature(clean_pair, y, min_freq)
        metrics['pair'] = pair_name
        results.append(metrics)

    report = pd.DataFrame(results)
    report = report.sort_values(
        by=['mutual_info', 'singleton_ratio'], ascending=[False, True]
    ).reset_index(drop=True)
    cols_order = ['pair'] + [c for c in report.columns if c != 'pair']
    report = report[cols_order]

    return report.head(top_n) if top_n else report
