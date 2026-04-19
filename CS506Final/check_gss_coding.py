import pandas as pd
import numpy as np

print("="*70)
print("CHECKING GSS VARIABLE CODING")
print("="*70)

# Load only the columns we need
columns_needed = [
    'GENERAL HAPPINESS',
    'HOW OFTEN R ATTENDS RELIGIOUS SERVICES',
    'SPEND EVENING WITH FRIENDS',
    'THINK OF SELF AS LIBERAL OR CONSERVATIVE'
]

print("\nLoading data...")
df = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv', low_memory=False)

# Find and load the actual columns
happiness_col = None
attend_col = None
socfrend_col = None
polviews_col = None

for col in df.columns:
    col_upper = col.strip().upper()
    if 'GENERAL HAPPINESS' in col_upper and '_LABELS' not in col_upper:
        happiness_col = col
    elif 'HOW OFTEN R ATTENDS RELIGIOUS SERVICES' in col_upper and '_LABELS' not in col_upper:
        attend_col = col
    elif 'SPEND EVENING WITH FRIENDS' in col_upper and '_LABELS' not in col_upper:
        socfrend_col = col
    elif 'THINK OF SELF AS LIBERAL OR CONSERVATIVE' in col_upper and '_LABELS' not in col_upper:
        polviews_col = col

print(f"\nFound columns:")
print(f"  Happiness: {happiness_col}")
print(f"  Attend: {attend_col}")
print(f"  Socfrend: {socfrend_col}")
print(f"  Polviews: {polviews_col}")

# Check for label columns too
happiness_label_col = happiness_col + '_labels' if happiness_col else None
attend_label_col = attend_col + '_labels' if attend_col else None
socfrend_label_col = socfrend_col + '_labels' if socfrend_col else None
polviews_label_col = polviews_col + '_labels' if polviews_col else None

# Convert to numeric
happiness_numeric = pd.to_numeric(df[happiness_col], errors='coerce')
attend_numeric = pd.to_numeric(df[attend_col], errors='coerce') if attend_col else None
socfrend_numeric = pd.to_numeric(df[socfrend_col], errors='coerce') if socfrend_col else None
polviews_numeric = pd.to_numeric(df[polviews_col], errors='coerce') if polviews_col else None

print("\n" + "="*70)
print("1. HAPPINESS CODING")
print("="*70)
print(f"\nUnique values: {sorted(happiness_numeric.dropna().unique())}")
print(f"\nValue counts:")
print(happiness_numeric.value_counts().sort_index())

# Try to show labels if available
if happiness_label_col in df.columns:
    print(f"\nWith labels:")
    temp_df = df[[happiness_col, happiness_label_col]].drop_duplicates().dropna()
    temp_df = temp_df.sort_values(happiness_col)
    for _, row in temp_df.iterrows():
        print(f"  {row[happiness_col]} = {row[happiness_label_col]}")

print("\n" + "="*70)
print("2. ATTEND (Religious Services) CODING")
print("="*70)
if attend_col:
    print(f"\nUnique values: {sorted(attend_numeric.dropna().unique())}")
    print(f"\nValue counts:")
    print(attend_numeric.value_counts().sort_index())

    if attend_label_col in df.columns:
        print(f"\nWith labels:")
        temp_df = df[[attend_col, attend_label_col]].drop_duplicates().dropna()
        temp_df = temp_df.sort_values(attend_col)
        for _, row in temp_df.head(20).iterrows():
            print(f"  {row[attend_col]} = {row[attend_label_col]}")
else:
    print("Column not found!")

print("\n" + "="*70)
print("3. SOCFREND (Social Evenings) CODING")
print("="*70)
if socfrend_col:
    print(f"\nUnique values: {sorted(socfrend_numeric.dropna().unique())}")
    print(f"\nValue counts:")
    print(socfrend_numeric.value_counts().sort_index())

    if socfrend_label_col in df.columns:
        print(f"\nWith labels:")
        temp_df = df[[socfrend_col, socfrend_label_col]].drop_duplicates().dropna()
        temp_df = temp_df.sort_values(socfrend_col)
        for _, row in temp_df.head(20).iterrows():
            print(f"  {row[socfrend_col]} = {row[socfrend_label_col]}")
else:
    print("Column not found!")

print("\n" + "="*70)
print("4. POLVIEWS (Political Views) CODING")
print("="*70)
if polviews_col:
    print(f"\nUnique values: {sorted(polviews_numeric.dropna().unique())}")
    print(f"\nValue counts:")
    print(polviews_numeric.value_counts().sort_index())

    if polviews_label_col in df.columns:
        print(f"\nWith labels:")
        temp_df = df[[polviews_col, polviews_label_col]].drop_duplicates().dropna()
        temp_df = temp_df.sort_values(polviews_col)
        for _, row in temp_df.head(20).iterrows():
            print(f"  {row[polviews_col]} = {row[polviews_label_col]}")
else:
    print("Column not found!")

print("\n" + "="*70)
print("5. CORRELATION INTERPRETATION")
print("="*70)

# Show some cross-tabulations
if attend_col:
    print("\n--- HAPPINESS vs ATTEND ---")
    crosstab = pd.crosstab(happiness_numeric, attend_numeric, margins=True)
    print(crosstab.head(10))

if socfrend_col:
    print("\n--- HAPPINESS vs SOCFREND ---")
    crosstab = pd.crosstab(happiness_numeric, socfrend_numeric, margins=True)
    print(crosstab.head(10))

if polviews_col:
    print("\n--- HAPPINESS vs POLVIEWS ---")
    crosstab = pd.crosstab(happiness_numeric, polviews_numeric, margins=True)
    print(crosstab.head(10))

print("\n" + "="*70)
print("DONE!")
print("="*70)
