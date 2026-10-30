
def create_clean_pair(col1, col2):
    res_dept_raw = col1.astype(str) + '_' + col2.astype(str)
    counts = res_dept_raw.value_counts()
    return res_dept_raw.where(res_dept_raw.map(counts) >= 3, 'OTHER')


def create_frequency_cols(df_x, list_columns):
    freq_cols = []
    for col in list_columns:
        freq_col_name = f'{col}_freq'
        df_x[freq_col_name] = df_x[col].map(df_x[col].value_counts())
        freq_cols.append(freq_col_name)
    return freq_cols
