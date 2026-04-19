import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

print("="*70)
print("GSS HAPPINESS ANALYSIS - PEARSON CORRELATION")
print("="*70)

start_time = time()

# Step 1: Load column names
print("\n[1/4] Loading column names...")
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

print(f"   Found {len(columns_to_load)} columns")

# Step 2: Load data
print("\n[2/4] Loading data...")
df = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv',
                 usecols=columns_to_load,
                 low_memory=False)
print(f"   Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# Step 3: Convert to numeric
print("\n[3/4] Converting to numeric and calculating correlations...")
df_numeric = df.apply(pd.to_numeric, errors='coerce')
happiness_col = column_mapping['HAPPY']
happiness_values = df_numeric[happiness_col].values
happiness_mask = ~np.isnan(happiness_values)

# Calculate both Pearson and Spearman for comparison
pearson_results = []
spearman_results = []

for factor_name, factor_col in column_mapping.items():
    if factor_name == 'HAPPY':
        continue

    factor_values = df_numeric[factor_col].values
    valid_mask = happiness_mask & ~np.isnan(factor_values)

    valid_happiness = happiness_values[valid_mask]
    valid_factor = factor_values[valid_mask]

    n = len(valid_happiness)

    if n > 10:
        # Pearson correlation (linear relationship)
        pearson_corr, pearson_pval = stats.pearsonr(valid_happiness, valid_factor)

        # Spearman for comparison
        spearman_corr, spearman_pval = stats.spearmanr(valid_happiness, valid_factor)

        pearson_results.append({
            'Factor': factor_name,
            'Correlation': pearson_corr,
            'P-value': pearson_pval,
            'N': n,
            'Significant': pearson_pval < 0.05
        })

        spearman_results.append({
            'Factor': factor_name,
            'Correlation': spearman_corr,
            'P-value': spearman_pval,
            'N': n,
            'Significant': spearman_pval < 0.05
        })

# Create DataFrames
pearson_df = pd.DataFrame(pearson_results)
spearman_df = pd.DataFrame(spearman_results)

# Sort by absolute correlation
pearson_df = pearson_df.iloc[np.argsort(-np.abs(pearson_df['Correlation'].values))]
spearman_df = spearman_df.iloc[np.argsort(-np.abs(spearman_df['Correlation'].values))]

print("\n" + "="*70)
print("PEARSON CORRELATION RESULTS")
print("="*70)
for _, row in pearson_df.iterrows():
    sig_mark = "✓" if row['Significant'] else "✗"
    print(f"{sig_mark} {row['Factor']:10s}: r={row['Correlation']:7.4f}, "
          f"p={row['P-value']:.2e}, N={row['N']:,}")

# Step 4: Create comparison visualization
print("\n[4/4] Creating visualization...")

fig = plt.figure(figsize=(20, 8))
gs = fig.add_gridspec(1, 3, wspace=0.25)

# Merge data for comparison
comparison_df = pearson_df.merge(spearman_df, on='Factor', suffixes=('_pearson', '_spearman'))

# Sort by Pearson correlation for consistent ordering
comparison_df = comparison_df.iloc[np.argsort(-np.abs(comparison_df['Correlation_pearson'].values))]

y_pos = np.arange(len(comparison_df))

# Plot 1: Pearson Correlation
ax1 = fig.add_subplot(gs[0, 0])
pearson_colors = np.where(comparison_df['Significant_pearson'].values, '#e74c3c', '#c0392b')

bars1 = ax1.barh(y_pos, comparison_df['Correlation_pearson'].values,
                 color=pearson_colors, alpha=0.85, edgecolor='black', linewidth=1.5)

ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(comparison_df['Factor'].values, fontsize=11)
ax1.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Factor', fontsize=12, fontweight='bold')
ax1.set_title('Pearson Correlation\n(Linear Relationship)',
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)

# Add values
for i, corr in enumerate(comparison_df['Correlation_pearson'].values):
    x_offset = 0.012 if corr > 0 else -0.012
    ha = 'left' if corr > 0 else 'right'
    ax1.text(corr + x_offset, i, f'{corr:.3f}',
             va='center', ha=ha, fontsize=9, fontweight='bold')

# Plot 2: Spearman Correlation
ax2 = fig.add_subplot(gs[0, 1])
spearman_colors = np.where(comparison_df['Significant_spearman'].values, '#27ae60', '#229954')

bars2 = ax2.barh(y_pos, comparison_df['Correlation_spearman'].values,
                 color=spearman_colors, alpha=0.85, edgecolor='black', linewidth=1.5)

ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(comparison_df['Factor'].values, fontsize=11)
ax2.set_xlabel('Spearman Correlation Coefficient (ρ)', fontsize=12, fontweight='bold')
ax2.set_title('Spearman Correlation\n(Rank-Based Relationship)',
              fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)

# Add values
for i, corr in enumerate(comparison_df['Correlation_spearman'].values):
    x_offset = 0.012 if corr > 0 else -0.012
    ha = 'left' if corr > 0 else 'right'
    ax2.text(corr + x_offset, i, f'{corr:.3f}',
             va='center', ha=ha, fontsize=9, fontweight='bold')

# Plot 3: Side-by-side comparison
ax3 = fig.add_subplot(gs[0, 2])

# Create grouped bars
bar_width = 0.35
y_pos_offset = np.arange(len(comparison_df))

bars_p = ax3.barh(y_pos_offset - bar_width/2, comparison_df['Correlation_pearson'].values,
                  bar_width, label='Pearson', color='#e74c3c', alpha=0.85,
                  edgecolor='black', linewidth=1)

bars_s = ax3.barh(y_pos_offset + bar_width/2, comparison_df['Correlation_spearman'].values,
                  bar_width, label='Spearman', color='#27ae60', alpha=0.85,
                  edgecolor='black', linewidth=1)

ax3.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
ax3.set_yticks(y_pos_offset)
ax3.set_yticklabels(comparison_df['Factor'].values, fontsize=11)
ax3.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
ax3.set_title('Direct Comparison\n(Red=Pearson, Green=Spearman)',
              fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=11, loc='lower right')
ax3.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)

# Add difference annotations
for i, (p_corr, s_corr) in enumerate(zip(comparison_df['Correlation_pearson'].values,
                                          comparison_df['Correlation_spearman'].values)):
    diff = s_corr - p_corr
    x_pos = max(abs(p_corr), abs(s_corr)) + 0.02
    color = 'green' if abs(diff) < 0.02 else 'orange' if abs(diff) < 0.05 else 'red'
    ax3.text(x_pos, i, f'Δ={diff:+.3f}', va='center', ha='left',
             fontsize=8, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('gss_happiness_pearson_comparison.png', dpi=300,
            bbox_inches='tight', facecolor='white')

print(f"   Visualization saved!")

# Create detailed comparison table
print("\n" + "="*70)
print("PEARSON vs SPEARMAN COMPARISON")
print("="*70)
print(f"{'Factor':<10} {'Pearson':>8} {'Spearman':>8} {'Difference':>10} {'Interpretation'}")
print("-" * 70)
for _, row in comparison_df.iterrows():
    diff = row['Correlation_spearman'] - row['Correlation_pearson']

    if abs(diff) < 0.02:
        interp = "Similar"
    elif abs(diff) < 0.05:
        interp = "Small diff"
    else:
        interp = "Notable diff"

    print(f"{row['Factor']:<10} {row['Correlation_pearson']:>8.4f} "
          f"{row['Correlation_spearman']:>8.4f} {diff:>+10.4f} {interp}")

# Save results
comparison_df.to_csv('gss_happiness_pearson_vs_spearman.csv', index=False)
print("\n✓ Results saved to: gss_happiness_pearson_vs_spearman.csv")

total_time = time() - start_time
print(f"\n✓ Analysis completed in {total_time:.2f}s")
print("="*70)
