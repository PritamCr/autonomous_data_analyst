def run_eda(df):

    report = {}

    report["shape"] = df.shape
    report["columns"] = df.columns.tolist()
    report["missing_values"] = df.isnull().sum()
    report["data_types"] = df.dtypes
    report["summary"] = df.describe()

    return report