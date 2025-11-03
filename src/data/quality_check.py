import pandas as pd
import numpy as np

def check_missing_values(df):
    """Return a styled table of missing values."""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = (
        pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        .query('`Missing Count` > 0')
        .sort_values('Missing %', ascending=False)
        .style.background_gradient(subset=['Missing %'], cmap='Reds')
        .format({'Missing %': '{:.2f}%'})
    )
    return missing_df


def check_duplicates(df, subset=None):
    """Return a small summary DataFrame of duplicates."""
    total_dup = df.duplicated(subset=subset).sum()
    pct_dup = 100 * total_dup / len(df)
    summary = pd.DataFrame({
        'Total Rows': [len(df)],
        'Duplicate Rows': [total_dup],
        'Duplicate %': [pct_dup]
    })
    return summary.style.format({'Duplicate %': '{:.2f}%'})


def check_data_types(df):
    """Summarize column types, non-null counts, and unique values."""
    summary = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str).values,
        'Non-Null Count': df.count().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    return summary.style.hide(axis="index").background_gradient(subset=['Unique Values'], cmap='Blues')


def detect_outliers(df, column, method='iqr', threshold=1.5):
    """Detect outliers in a numeric column."""
    if column not in df.columns:
        raise ValueError(f"Column {column} not found")

    data = df[column].dropna()

    if method == 'iqr':
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
        outliers = data[(data < lower) | (data > upper)]

    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = data[z_scores > threshold]
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

    return pd.DataFrame({
        'Total Points': [len(data)],
        'Outliers': [len(outliers)],
        'Outlier %': [100 * len(outliers) / len(data)],
        'Lower Bound': [lower],
        'Upper Bound': [upper]
    }).style.format({'Outlier %': '{:.2f}%'})

def get_numeric_cols_count(df):
    return len(df.select_dtypes(include=[np.number]).columns)

def get_categorical_cols_count(df):
    return len(df.select_dtypes(exclude=[np.number]).columns)

def get_avg_row_size_bytes(df):
    mem_bytes = int(df.memory_usage(deep=True).sum())
    return round(mem_bytes / max(len(df), 1), 1)

def get_nonnull_pct_avg(df):
    return round(df.notnull().mean().mean() * 100, 2)

def get_unique_pct_avg(df):
    n = len(df)
    return round(df.nunique().mean() / n * 100, 2) if n else 0

def get_min_numeric(df):
    num = df.select_dtypes(include=[np.number])
    return num.min().min() if not num.empty else None

def get_max_numeric(df):
    num = df.select_dtypes(include=[np.number])
    return num.max().max() if not num.empty else None

def generate_report(df):
    """Generate a compact summary DataFrame of the dataset."""
    info = {
        'Rows': len(df),
        'Columns': df.shape[1],
        'Memory Usage (MB)': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'Duplicate %': round(100 * df.duplicated().sum() / len(df), 2),
        'Missing % (overall)': round(100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1]), 2),
        'Numeric Columns': len(df.select_dtypes(include=[np.number]).columns),
        'Categorical Columns': len(df.select_dtypes(exclude=[np.number]).columns)
    }
    summary = pd.DataFrame(info, index=['Summary'])
    return summary.style.format({
        'Memory Usage (MB)': '{:.2f}',
        'Duplicate %': '{:.2f}%',
        'Missing % (overall)': '{:.2f}%'
    })


def check_data_quality_on_datasets(df_list, names=None):
    """Compare overall data quality across multiple datasets."""
    reports = []
    for i, df in enumerate(df_list):
        name = names[i] if names and i < len(names) else f"Dataset_{i+1}"
        num = df.select_dtypes(include=[np.number])
        total_vals = len(df) * df.shape[1]
        info = {
            'Dataset': name,
            'Rows': len(df),
            'Columns': df.shape[1],
            'Missing %': round(100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1]), 2),
            'Duplicate %': round(100 * df.duplicated().sum() / len(df), 2),
            'Memory (MB)': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'Numeric Cols': len(num.columns),
            'Categorical Cols': df.shape[1] - len(num.columns),
            'Non-Null % (avg)': round(df.notnull().mean().mean() * 100, 2),
            'Unique % (avg)': round((df.nunique().mean() / len(df) * 100), 2) if len(df) else 0,
            'Min Numeric': (num.min().min() if not num.empty else None),
            'Max Numeric': (num.max().max() if not num.empty else None),
        }
        reports.append(info)
    return pd.DataFrame(reports).sort_values("Rows", ascending=False)

