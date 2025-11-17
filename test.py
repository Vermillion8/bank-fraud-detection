import pandas as pd
import numpy as np

# Sample DataFrame
data = {'A': [1, 2, 2, 3, np.nan],
        'B': ['apple', 'banana', 'apple', 'orange', 'banana'],
        'C': [True, False, True, True, False]}
df = pd.DataFrame(data)

# Count unique values in a specific column (Series)
unique_in_A = df['A'].nunique()
print(f"Unique values in column 'A': {unique_in_A}")

# Count unique values in the entire DataFrame (per column)
unique_counts_df = df.nunique()
print("\nUnique counts per column:")
print(unique_counts_df)

# Count unique values including NaN
unique_in_A_with_nan = df['A'].nunique(dropna=False)
print(f"\nUnique values in column 'A' (including NaN): {unique_in_A_with_nan}")

# Count unique values per row
unique_counts_per_row = df.nunique(axis=1)
print("\nUnique counts per row:")
print(unique_counts_per_row)
