import pandas as pd
import numpy as np

def check_missing_values(df):
    """Report missing values in the dataset."""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })

    return missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Percentage', ascending=False
    )

def check_duplicates(df, subset=None):
    """Check for duplicate rows."""
    duplicates = df.duplicated(subset=subset).sum()
    return {
        'total_duplicates': duplicates,
        'percentage': 100 * duplicates / len(df)
    }

def check_data_types(df):
    """Report data types of all columns."""
    return pd.DataFrame({
        'Column': df.columns,
        'Data_Type': df.dtypes.values,
        'Non_Null_Count': df.count().values,
        'Unique_Values': [df[col].nunique() for col in df.columns]
    })

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """Detect outliers in a numeric column."""
    if column not in df.columns:
        raise ValueError(f"Column {column} not found")

    data = df[column].dropna()

    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]

    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = data[z_scores > threshold]

    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

    return {
        'num_outliers': len(outliers),
        'percentage': 100 * len(outliers) / len(data),
        'outlier_values': outliers.describe()
    }

def generate_report(df):
    """Generate comprehensive data quality report."""
    report = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicates': check_duplicates(df),
        'data_types': check_data_types(df).to_dict()
    }
    return report
