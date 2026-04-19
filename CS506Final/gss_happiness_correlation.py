import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the GSS data
print("Loading GSS data...")
df = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv')

print(f"Dataset shape: {df.shape}")
print("\nSearching for column names containing key terms...")

# Search for exact column names - the GSS data has very long column names
# Let's search more carefully for exact matches
factor_searches = {
    'MARITAL': 'MARITAL STATUS',
    'INCOME': 'TOTAL FAMILY INCOME',
    'WRKSTAT': 'LABOR FORCE STATUS',
    'DEGREE': 'RS HIGHEST DEGREE',  # R's highest degree
    'ATTEND': 'HOW OFTEN R ATTENDS RELIGIOUS SERVICES',
    'TRUST': 'CAN PEOPLE BE TRUSTED',
    'POLVIEWS': 'THINK OF SELF AS LIBERAL OR CONSERVATIVE',
    'HEALTH': 'CONDITION OF HEALTH',
    'SOCFREND': 'SPEND EVENING WITH FRIENDS',
    'SIZE': 'SIZE OF PLACE IN 1000S'
}

# Find happiness column - looking for "GENERAL HAPPINESS"
happiness_col = None
for col in df.columns:
    if col.strip().upper() == 'GENERAL HAPPINESS':
        happiness_col = col
        break

if not happiness_col:
    print("Could not find 'GENERAL HAPPINESS' column")
    exit(1)

print(f"\nUsing happiness column: {happiness_col}")

# Find the factor columns by searching for best matches
factor_cols = {}
for key, search_term in factor_searches.items():
    best_match = None
    for col in df.columns:
        col_clean = col.strip().upper()
        if '_LABELS' not in col_clean:
            if search_term.upper() in col_clean or col_clean in search_term.upper():
                # Check for more exact match
                if search_term.upper() == col_clean or col_clean.startswith(search_term.upper()):
                    best_match = col
                    break
                elif not best_match:
                    best_match = col

    if best_match:
        factor_cols[key] = best_match
        print(f"{key}: {best_match}")
    else:
        print(f"{key}: NOT FOUND")

# Calculate correlations and p-values
results = []

for factor_name, factor_col in factor_cols.items():
    # Convert to numeric, coercing errors to NaN
    happy_numeric = pd.to_numeric(df[happiness_col], errors='coerce')
    factor_numeric = pd.to_numeric(df[factor_col], errors='coerce')

    # Create a DataFrame with both columns and drop any rows with NaN
    subset = pd.DataFrame({
        'happiness': happy_numeric,
        'factor': factor_numeric
    }).dropna()

    if len(subset) > 10:  # Need at least 10 data points for meaningful correlation
        try:
            # Calculate Spearman correlation (good for ordinal data)
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
            print(f"  Significant at α=0.05: {pval < 0.05}")
        except Exception as e:
            print(f"\n{factor_name}: Error calculating correlation - {e}")
    else:
        print(f"\n{factor_name}: Insufficient data ({len(subset)} points)")

# Create DataFrame with results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Correlation', key=abs, ascending=False)

print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
print(results_df.to_string(index=False))

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Correlation coefficients with significance markers
colors = ['green' if sig else 'red' for sig in results_df['Significant']]
bars = ax1.barh(results_df['Factor'], results_df['Correlation'], color=colors, alpha=0.7)

ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Spearman Correlation Coefficient', fontsize=12)
ax1.set_ylabel('Factor', fontsize=12)
ax1.set_title('Correlation with Happiness\n(Green=Significant p<0.05, Red=Not Significant)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add correlation values on bars
for i, (idx, row) in enumerate(results_df.iterrows()):
    x_pos = row['Correlation'] + (0.01 if row['Correlation'] > 0 else -0.01)
    ha = 'left' if row['Correlation'] > 0 else 'right'
    ax1.text(x_pos, i, f"{row['Correlation']:.3f}", va='center', ha=ha, fontsize=9)

# Plot 2: -log10(p-value) to show significance
results_df['neg_log_pval'] = -np.log10(results_df['P-value'])
significance_threshold = -np.log10(0.05)

colors2 = ['green' if sig else 'red' for sig in results_df['Significant']]
bars2 = ax2.barh(results_df['Factor'], results_df['neg_log_pval'], color=colors2, alpha=0.7)

ax2.axvline(x=significance_threshold, color='blue', linestyle='--', linewidth=2, label='p=0.05 threshold')
ax2.set_xlabel('-log10(p-value)', fontsize=12)
ax2.set_ylabel('Factor', fontsize=12)
ax2.set_title('Statistical Significance of Correlations\n(Higher values = more significant)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Add p-values as text
for i, (idx, row) in enumerate(results_df.iterrows()):
    x_pos = row['neg_log_pval'] + 1
    if row['P-value'] < 0.001:
        pval_text = 'p<0.001'
    else:
        pval_text = f'p={row["P-value"]:.3f}'
    ax2.text(x_pos, i, pval_text, va='center', ha='left', fontsize=9)

plt.tight_layout()
plt.savefig('gss_happiness_correlations.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'gss_happiness_correlations.png'")

# Save results to CSV
results_df.to_csv('gss_happiness_correlation_results.csv', index=False)
print("Results saved to 'gss_happiness_correlation_results.csv'")

plt.show()
