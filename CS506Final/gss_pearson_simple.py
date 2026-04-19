import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from time import time

print("Loading GSS data...")

start_time = time()

# Load column names
df_cols = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv', nrows=0)
all_columns = list(df_cols.columns)

columns_to_load = []
column_mapping = {}

# Find happiness column
for col in all_columns:
    if col.strip().upper() == 'GENERAL HAPPINESS':
        columns_to_load.append(col)
        column_mapping['HAPPY'] = col
        break

# Find factor columns
search_terms = {
    'MARITAL': 'MARITAL STATUS',
    'INCOME': 'TOTAL FAMILY INCOME',
    'WRKSTAT': 'LABOR FORCE STATUS',
    'DEGREE': 'RS HIGHEST DEGREE',
    'ATTEND': 'HOW OFTEN R ATTENDS RELIGIOUS SERVICES',
    'TRUST': 'CAN PEOPLE BE TRUSTED',
    'POLVIEWS': 'THINK OF SELF AS LIBERAL OR CONSERVATIVE',
    'HEALTH': 'CONDITION OF HEALTH',
    'SOCFREND': 'SPEND EVENING WITH FRIENDS',
    'SIZE': 'SIZE OF PLACE IN 1000S'
}

for key, search_term in search_terms.items():
    for col in all_columns:
        col_upper = col.strip().upper()
        if '_LABELS' not in col_upper:
            if search_term.upper() in col_upper or col_upper.startswith(search_term.upper()):
                columns_to_load.append(col)
                column_mapping[key] = col
                break

# Load data
df = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv',
                 usecols=columns_to_load,
                 low_memory=False)

# Convert to numeric
df_numeric = df.apply(pd.to_numeric, errors='coerce')
happiness_col = column_mapping['HAPPY']
happiness_values = df_numeric[happiness_col].values
happiness_mask = ~np.isnan(happiness_values)

# Calculate Pearson correlations
pearson_results = []

for factor_name, factor_col in column_mapping.items():
    if factor_name == 'HAPPY':
        continue

    factor_values = df_numeric[factor_col].values
    valid_mask = happiness_mask & ~np.isnan(factor_values)

    valid_happiness = happiness_values[valid_mask]
    valid_factor = factor_values[valid_mask]

    n = len(valid_happiness)

    if n > 10:
        pearson_corr, pearson_pval = stats.pearsonr(valid_happiness, valid_factor)

        pearson_results.append({
            'Factor': factor_name,
            'Correlation': pearson_corr,
            'P-value': pearson_pval,
            'N': n
        })

# Create DataFrame
pearson_df = pd.DataFrame(pearson_results)
pearson_df = pearson_df.iloc[np.argsort(-np.abs(pearson_df['Correlation'].values))]

print("\nPearson Correlation Results:")
for _, row in pearson_df.iterrows():
    print(f"{row['Factor']:10s}: r={row['Correlation']:7.4f}")

# Create simple graph
plt.figure(figsize=(10, 6))

y_pos = np.arange(len(pearson_df))

# Simple blue bars, no grid
plt.barh(y_pos, pearson_df['Correlation'].values, color='blue')

# Simple labels
plt.yticks(y_pos, pearson_df['Factor'].values)
plt.xlabel('r')
plt.title('Pearson Correlation')

# Add a line at x=0
plt.axvline(x=0, color='black', linewidth=1)

# Save
plt.tight_layout()
plt.savefig('gss_pearson_simple.png', dpi=150, bbox_inches='tight', facecolor='white')

total_time = time() - start_time
print(f"\nDone! Graph saved as 'gss_pearson_simple.png'")
print(f"Time: {total_time:.2f}s")
