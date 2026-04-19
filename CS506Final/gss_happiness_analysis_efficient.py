import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading column names first...")
# First, just get the column names without loading all data
df_cols = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv', nrows=0)
all_columns = list(df_cols.columns)

print(f"Total columns in dataset: {len(all_columns)}")

# Find the columns we need
columns_to_load = []
column_mapping = {}

# Happiness column
for col in all_columns:
    if col.strip().upper() == 'GENERAL HAPPINESS':
        columns_to_load.append(col)
        column_mapping['HAPPY'] = col
        print(f"Found HAPPINESS: {col}")
        break

# Factor columns - exact search terms from the user
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
    found = False
    for col in all_columns:
        col_upper = col.strip().upper()
        if '_LABELS' not in col_upper:
            # Look for exact or close match
            if search_term.upper() in col_upper or col_upper.startswith(search_term.upper()):
                columns_to_load.append(col)
                column_mapping[key] = col
                print(f"Found {key}: {col}")
                found = True
                break
    if not found:
        print(f"WARNING: Could not find column for {key}")

print(f"\nLoading {len(columns_to_load)} columns...")

# Now load only the columns we need
df = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv', usecols=columns_to_load, low_memory=False)

print(f"Loaded data shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Calculate correlations
results = []

happiness_col = column_mapping['HAPPY']

for factor_name, factor_col in column_mapping.items():
    if factor_name == 'HAPPY':
        continue

    # Convert to numeric
    happy_numeric = pd.to_numeric(df[happiness_col], errors='coerce')
    factor_numeric = pd.to_numeric(df[factor_col], errors='coerce')

    # Create subset without NaN
    subset = pd.DataFrame({
        'happiness': happy_numeric,
        'factor': factor_numeric
    }).dropna()

    if len(subset) > 10:
        try:
            # Spearman correlation (robust for ordinal data)
            corr, pval = stats.spearmanr(subset['happiness'], subset['factor'])

            results.append({
                'Factor': factor_name,
                'Correlation': corr,
                'P-value': pval,
                'N': len(subset),
                'Significant': pval < 0.05
            })

            print(f"\n{factor_name}:")
            print(f"  Correlation: {corr:.4f}")
            print(f"  P-value: {pval:.4e}")
            print(f"  Sample size: {len(subset)}")
            print(f"  Significant (p<0.05): {pval < 0.05}")

        except Exception as e:
            print(f"\n{factor_name}: ERROR - {e}")
    else:
        print(f"\n{factor_name}: Insufficient data ({len(subset)} points)")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Correlation', key=abs, ascending=False)

print("\n" + "="*70)
print("SUMMARY OF CORRELATIONS WITH HAPPINESS")
print("="*70)
print(results_df.to_string(index=False))

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Correlation coefficients with significance coloring
colors = ['#2ecc71' if sig else '#e74c3c' for sig in results_df['Significant']]
bars = ax1.barh(results_df['Factor'], results_df['Correlation'], color=colors, alpha=0.8, edgecolor='black')

ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Spearman Correlation Coefficient', fontsize=13, fontweight='bold')
ax1.set_ylabel('Factor', fontsize=13, fontweight='bold')
ax1.set_title('Correlation with Happiness\n(Green = Significant at p<0.05, Red = Not Significant)',
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add correlation values on bars
for i, (idx, row) in enumerate(results_df.iterrows()):
    x_pos = row['Correlation'] + (0.015 if row['Correlation'] > 0 else -0.015)
    ha = 'left' if row['Correlation'] > 0 else 'right'
    ax1.text(x_pos, i, f"{row['Correlation']:.3f}", va='center', ha=ha,
             fontsize=10, fontweight='bold')

# Plot 2: Statistical significance (-log10 p-value)
results_df['neg_log_pval'] = -np.log10(results_df['P-value'])
significance_threshold = -np.log10(0.05)

bars2 = ax2.barh(results_df['Factor'], results_df['neg_log_pval'], color=colors, alpha=0.8, edgecolor='black')

ax2.axvline(x=significance_threshold, color='#3498db', linestyle='--', linewidth=3,
            label=f'p=0.05 threshold', alpha=0.8)
ax2.set_xlabel('-log₁₀(p-value)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Factor', fontsize=13, fontweight='bold')
ax2.set_title('Statistical Significance of Correlations\n(Higher values = More significant)',
              fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Add p-value labels
for i, (idx, row) in enumerate(results_df.iterrows()):
    x_pos = row['neg_log_pval'] + max(results_df['neg_log_pval']) * 0.03
    if row['P-value'] < 0.001:
        pval_text = 'p<0.001'
    elif row['P-value'] < 0.01:
        pval_text = f'p={row["P-value"]:.3f}'
    else:
        pval_text = f'p={row["P-value"]:.2f}'
    ax2.text(x_pos, i, pval_text, va='center', ha='left', fontsize=9)

plt.tight_layout()
plt.savefig('gss_happiness_correlations.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Visualization saved as 'gss_happiness_correlations.png'")

# Save results to CSV
results_df.to_csv('gss_happiness_correlation_results.csv', index=False)
print("✓ Results saved to 'gss_happiness_correlation_results.csv'")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
