import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from time import time
print("Loading")
start_time = time()
tmp = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv', nrows=0)
cols =list(tmp.columns )
to_load = []
cmap = {}
# find  col
for c in cols:
    if c.strip().upper() == 'GENERAL HAPPINESS':
        to_load.append(c)
        cmap['HAPPY'] = c
        break
targets = {'MARITAL': 'MARITAL STATUS', 'INCOME': 'TOTAL FAMILY INCOME', 'WRKSTAT': 'LABOR FORCE STATUS', 'DEGREE': 'RS HIGHEST DEGREE', 'ATTEND': 'HOW OFTEN R ATTENDS RELIGIOUS SERVICES', 'TRUST': 'CAN PEOPLE BE TRUSTED', 'POLVIEWS': 'THINK OF SELF AS LIBERAL OR CONSERVATIVE', 'HEALTH': 'CONDITION OF HEALTH', 'SOCFREND': 'SPEND EVENING WITH FRIENDS', 'SIZE': 'SIZE OF PLACE IN 1000S'}

# Look for the terms we already identified
for key, term in targets.items():
    for c in cols:
        cu = c.strip().upper()
        if '_LABELS' not in cu:
            if term.upper() in cu or cu.startswith(term.upper()):
                to_load.append(c)
                cmap[key] = c
                break
# Load file, which will take a while
df = pd.read_csv('GSS_Data_CSV_CodeBook/gss.csv', usecols=to_load, low_memory=False)
df2 = df.apply(pd.to_numeric, errors='coerce')
hcol = cmap['HAPPY']
hvals = df2[hcol].values
hmask = ~np.isnan(hvals)
# output
out = []
for fname, fcol in cmap.items():
    if fname == 'HAPPY':
        continue
    fvals = df2[fcol].values
    mask = hmask & ~np.isnan(fvals)
    n = mask.sum()
    if n > 10:
        r, p = stats.pearsonr(hvals[mask], fvals[mask])
        out.append({'Factor': fname, 'Correlation': r, 'P-value': p, 'N': n})
# create data frame
pearson_df = pd.DataFrame(out)
pearson_df = pearson_df.iloc[np.argsort(-np.abs(pearson_df['Correlation'].values))]
# print results
print("\nPearson Correlation Results:")
for _, row in pearson_df.iterrows():
    print(f"{row['Factor']:10s}: r={row['Correlation']:7.4f}")
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(pearson_df))
plt.barh(y_pos, pearson_df['Correlation'].values, color='blue')
plt.yticks(y_pos, pearson_df['Factor'].values)
plt.xlabel('r')
plt.title('Pearson Correlation')
plt.axvline(x=0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig('gss_pearson_simple.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nDone! Graph saved as 'gss_pearson_simple.png'")
print(f"Time: {time()-start_time:.2f}s")
