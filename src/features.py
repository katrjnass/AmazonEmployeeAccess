import pandas as pd

DROP_COLS = ['ACTION', 'id', 'ROLE_CODE', 'ROLE_FAMILY']


def create_clean_pair(col1: pd.Series, col2: pd.Series) -> pd.Series:
    res_dept_raw = col1.astype(str) + '_' + col2.astype(str)
    counts = res_dept_raw.value_counts()
    return res_dept_raw.where(res_dept_raw.map(counts) >= 3, 'OTHER')


def create_frequency_cols(df_x: pd.DataFrame, list_columns: list[str]) -> list[str]:
    freq_cols = []
    for col in list_columns:
        freq_col_name = f'{col}_freq'
        df_x[freq_col_name] = df_x[col].map(df_x[col].value_counts())
        freq_cols.append(freq_col_name)
    return freq_cols


def prepare_features(df_x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build features. Counts are computed over all rows of df_x (train + test)."""
    df_x = df_x.drop(columns=[c for c in DROP_COLS if c in df_x.columns])

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

    return df_x, cat_cols


def build_train_test(
        train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Concatenate train and test for frequency counts, then split back. X_test is indexed by id."""
    y_train = train['ACTION']
    full = pd.concat([train.drop(columns='ACTION'), test.drop(columns='id')], ignore_index=True)
    full_x, cat_cols = prepare_features(full)

    n_train = len(train)
    X_train = full_x.iloc[:n_train]
    X_test = full_x.iloc[n_train:].set_axis(test['id'].values)
    return X_train, y_train, X_test, cat_cols
