def explain_missing(col, pct, method):
    return (
        f"The column '{col}' contains missing values in {pct:.2f}% of records. "
        f"{method} imputation was selected because it preserves the central tendency "
        "of the data and reduces the influence of extreme values."
    )

def explain_duplicates(count):
    return (
        f"The dataset contains {count} duplicate rows. "
        "Duplicate records can bias analysis and lead to incorrect conclusions, "
        "so removing them improves data reliability."
    )
