import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

print("="*70)
print("OPTIMIZED GSS HAPPINESS CORRELATION ANALYSIS")
print("="*70)

start_time = time()

# Step 1: Efficiently load only needed columns
print("\n[1/5] Loading column names...")
df_cols = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv', nrows=0)
all_columns = list(df_cols.columns)

# Find columns to load
columns_to_load = []
column_mapping = {}

# Happiness column
for col in all_columns:
    if col.strip().upper() == 'GENERAL HAPPINESS':
        columns_to_load.append(col)
        column_mapping['HAPPY'] = col
        break

# Factor columns
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

print(f"   Found {len(columns_to_load)} columns to analyze")

# Step 2: Load data with optimizations
print("\n[2/5] Loading data with optimizations...")
load_start = time()

# Load only needed columns with low_memory=False for better performance
df = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv',
                 usecols=columns_to_load,
                 low_memory=False)

load_time = time() - load_start
print(f"   Loaded {df.shape[0]:,} rows × {df.shape[1]} columns in {load_time:.2f}s")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Step 3: Vectorized numeric conversion
print("\n[3/5] Converting to numeric (vectorized)...")
convert_start = time()

happiness_col = column_mapping['HAPPY']

# Convert all columns to numeric at once (vectorized)
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Get happiness values once
happiness_values = df_numeric[happiness_col].values
happiness_mask = ~np.isnan(happiness_values)

convert_time = time() - convert_start
print(f"   Converted in {convert_time:.2f}s")

# Step 4: Vectorized correlation calculation
print("\n[4/5] Calculating correlations (vectorized)...")
corr_start = time()

results = []
factor_names = []
correlations = []
pvalues = []
sample_sizes = []

for factor_name, factor_col in column_mapping.items():
    if factor_name == 'HAPPY':
        continue

    # Get factor values
    factor_values = df_numeric[factor_col].values

    # Create combined mask for non-NaN values in both arrays
    valid_mask = happiness_mask & ~np.isnan(factor_values)

    # Extract valid values using boolean indexing (very fast)
    valid_happiness = happiness_values[valid_mask]
    valid_factor = factor_values[valid_mask]

    n = len(valid_happiness)

    if n > 10:
        # Spearman correlation using scipy (optimized C implementation)
        corr, pval = stats.spearmanr(valid_happiness, valid_factor)

        factor_names.append(factor_name)
        correlations.append(corr)
        pvalues.append(pval)
        sample_sizes.append(n)

# Create results DataFrame using vectorized operations
results_df = pd.DataFrame({
    'Factor': factor_names,
    'Correlation': correlations,
    'P-value': pvalues,
    'N': sample_sizes,
    'Significant': np.array(pvalues) < 0.05
})

# Sort by absolute correlation
results_df = results_df.iloc[np.argsort(-np.abs(results_df['Correlation'].values))]

corr_time = time() - corr_start
print(f"   Calculated {len(results_df)} correlations in {corr_time:.2f}s")

# Print results
print("\n" + "="*70)
print("CORRELATION RESULTS")
print("="*70)
for _, row in results_df.iterrows():
    sig_mark = "✓" if row['Significant'] else "✗"
    print(f"{sig_mark} {row['Factor']:10s}: r={row['Correlation']:7.4f}, "
          f"p={row['P-value']:.2e}, N={row['N']:,}")

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

# Step 5: Create visualization
print("\n[5/5] Creating visualization...")
viz_start = time()

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

# Color scheme
sig_colors = np.where(results_df['Significant'].values, '#27ae60', '#e74c3c')

# Plot 1: Correlation coefficients
ax1 = fig.add_subplot(gs[0, 0])
y_pos = np.arange(len(results_df))

bars = ax1.barh(y_pos, results_df['Correlation'].values,
                color=sig_colors, alpha=0.85, edgecolor='black', linewidth=1.5)

ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(results_df['Factor'].values, fontsize=11)
ax1.set_xlabel('Spearman Correlation Coefficient (ρ)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Factor', fontsize=13, fontweight='bold')
ax1.set_title('Correlation with Happiness\n(Green = p<0.05, Red = Not Significant)',
              fontsize=15, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
ax1.set_xlim(min(results_df['Correlation'].min() * 1.2, -0.02),
             max(results_df['Correlation'].max() * 1.2, 0.02))

# Add correlation values on bars
for i, corr in enumerate(results_df['Correlation'].values):
    x_offset = 0.015 if corr > 0 else -0.015
    ha = 'left' if corr > 0 else 'right'
    ax1.text(corr + x_offset, i, f'{corr:.3f}',
             va='center', ha=ha, fontsize=10, fontweight='bold')

# Add sample sizes
for i, n in enumerate(results_df['N'].values):
    ax1.text(0.98, i, f'n={n:,}', transform=ax1.get_yaxis_transform(),
             va='center', ha='right', fontsize=8, alpha=0.6,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Plot 2: Statistical significance
ax2 = fig.add_subplot(gs[0, 1])

# Calculate -log10(p-value), handling p=0
with np.errstate(divide='ignore'):
    neg_log_p = -np.log10(results_df['P-value'].values)
    # Cap very small p-values at 300 for visualization
    neg_log_p = np.minimum(neg_log_p, 300)

bars2 = ax2.barh(y_pos, neg_log_p, color=sig_colors,
                 alpha=0.85, edgecolor='black', linewidth=1.5)

# Significance threshold
threshold = -np.log10(0.05)
ax2.axvline(x=threshold, color='#3498db', linestyle='--',
            linewidth=3, label='p=0.05', alpha=0.8, zorder=5)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(results_df['Factor'].values, fontsize=11)
ax2.set_xlabel('-log₁₀(p-value)', fontsize=13, fontweight='bold')
ax2.set_title('Statistical Significance\n(Higher = More Significant)',
              fontsize=15, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)

# Add p-value labels
for i, (pval, neg_log) in enumerate(zip(results_df['P-value'].values, neg_log_p)):
    if pval == 0:
        label = 'p≈0'
    elif pval < 0.001:
        label = 'p<0.001'
    elif pval < 0.01:
        label = f'p={pval:.3f}'
    else:
        label = f'p={pval:.2f}'

    x_offset = max(neg_log_p) * 0.03
    ax2.text(neg_log + x_offset, i, label,
             va='center', ha='left', fontsize=9, alpha=0.8)

plt.tight_layout()

# Save with high quality
plt.savefig('gss_happiness_correlations_optimized.png',
            dpi=300, bbox_inches='tight', facecolor='white',
            edgecolor='none')

viz_time = time() - viz_start
print(f"   Created visualization in {viz_time:.2f}s")

# Save results
results_df.to_csv('gss_happiness_results_optimized.csv', index=False, float_format='%.6f')
print(f"   Saved results to CSV")

# Performance summary
total_time = time() - start_time
print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)
print(f"   Data loading:     {load_time:6.2f}s  ({load_time/total_time*100:5.1f}%)")
print(f"   Conversion:       {convert_time:6.2f}s  ({convert_time/total_time*100:5.1f}%)")
print(f"   Correlations:     {corr_time:6.2f}s  ({corr_time/total_time*100:5.1f}%)")
print(f"   Visualization:    {viz_time:6.2f}s  ({viz_time/total_time*100:5.1f}%)")
print(f"   " + "-"*50)
print(f"   TOTAL TIME:       {total_time:6.2f}s")
print("="*70)
print("\n✓ Analysis complete!")
print(f"✓ Graph: gss_happiness_correlations_optimized.png")
print(f"✓ Data:  gss_happiness_results_optimized.csv")
