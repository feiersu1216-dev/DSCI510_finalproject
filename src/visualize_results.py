# CORRELATION BAR CHART - ONLY NUMERICALS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/processed/kpop_physical_sales.csv")
numeric_df = df.select_dtypes(include=['float64', 'int64'])

results = []
for col in numeric_df.columns:
    if col == "sales":
        continue
    data = numeric_df[["sales", col]].dropna()
    if len(data) > 2:
        from scipy.stats import pearsonr
        r, p = pearsonr(data["sales"], data[col])
        results.append((col, r, p))
    else:
        results.append((col, np.nan, np.nan))

results_df = pd.DataFrame(results, columns=["Predictor", "Correlation", "P_value"])
results_df["abs_corr"] = results_df["Correlation"].abs()

def sig_color(p):
    if pd.isna(p):
        return "gray"
    elif p < 0.01:
        return "darkgreen"
    elif p < 0.05:
        return "green"
    elif p < 0.1:
        return "lightgreen"
    else:
        return "gray"

results_df["color"] = results_df["P_value"].apply(sig_color)
results_df = results_df.sort_values("abs_corr", ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(results_df["Predictor"], results_df["Correlation"], color=results_df["color"])

for bar, corr in zip(bars, results_df["Correlation"]):
    width = bar.get_width()
    plt.text(width + (0.02 if width >= 0 else -0.07), bar.get_y() + bar.get_height() / 2,
             f"{corr:.2f}", va='center', ha='left' if width >= 0 else 'right', fontsize=9)

plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Correlation Coefficient with Sales")
plt.title("Correlation Coefficients & Significance with Sales")

import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color='darkgreen', label='p < 0.01'),
    mpatches.Patch(color='green', label='p < 0.05'),
    mpatches.Patch(color='lightgreen', label='p < 0.1'),
    mpatches.Patch(color='gray', label='not significant / NA'),
]
plt.legend(handles=legend_elements, title="Significance Levels", loc='lower right')

plt.tight_layout()
plt.show()


# CORRELATION MATRIX
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv("data/processed/kpop_physical_sales.csv")
numeric_df = df.select_dtypes(include=['float64', 'int64'])

cols = numeric_df.columns
n = len(cols)

corr = numeric_df.corr(method='pearson')

pvals = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
for i in range(n):
    for j in range(n):
        if i == j:
            pvals.iloc[i, j] = 0.0
        else:
            valid = numeric_df[[cols[i], cols[j]]].dropna()
            if len(valid) > 2:
                _, p = pearsonr(valid[cols[i]], valid[cols[j]])
                pvals.iloc[i, j] = p
            else:
                pvals.iloc[i, j] = np.nan

def annot_func(c, p):
    if np.isnan(p):
        return f"{c:.2f}\n(p=NA)"
    else:
        return f"{c:.2f}\n(p={p:.3f})"

annot = np.empty(corr.shape, dtype=object)
for i in range(n):
    for j in range(n):
        annot[i, j] = annot_func(corr.iloc[i, j], pvals.iloc[i, j])

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=annot, fmt="", cmap="coolwarm", center=0,
            linewidths=0.5, linecolor='white')
plt.title("Correlation Matrix with p-values")
plt.tight_layout()
plt.show()


# TOTAL SALES BY PARENT COMPANY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/kpop_physical_sales.csv")
df_clean = df.dropna(subset=["parent_company", "sales"])

sales_summary = df_clean.groupby("parent_company")["sales"].sum().reset_index()
sales_summary = sales_summary.rename(columns={'sales': 'total_sales'})
sales_summary = sales_summary.sort_values(by="total_sales", ascending=False).reset_index(drop=True)
sales_summary['rank'] = sales_summary.index + 1

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=sales_summary, x='total_sales', y='parent_company', palette='viridis')

plt.title('Total Sales by Parent Company (Ranked)')
plt.xlabel('Total Sales')
plt.ylabel('Parent Company')

for i, row in sales_summary.iterrows():
    ax.text(row['total_sales'] + row['total_sales']*0.01, i,
            f'#{row["rank"]} | Total: {row["total_sales"]:.0f}',
            va='center', fontsize=9, color='black')

plt.tight_layout()
plt.show()


# MEDIAN SALES BY PARENT COMPANY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/kpop_physical_sales.csv")
df_clean = df.dropna(subset=["parent_company", "sales"])

sales_summary = df_clean.groupby("parent_company")["sales"].median().reset_index()
sales_summary = sales_summary.rename(columns={'sales': 'median_sales'})
sales_summary = sales_summary.sort_values(by="median_sales", ascending=False).reset_index(drop=True)
sales_summary['rank'] = sales_summary.index + 1

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=sales_summary, x='median_sales', y='parent_company', palette='viridis')

plt.title('Median Sales by Parent Company (Ranked)')
plt.xlabel('Median Sales')
plt.ylabel('Parent Company')

for i, row in sales_summary.iterrows():
    ax.text(row['median_sales'] + row['median_sales']*0.01, i,
            f'#{row["rank"]} | Median: {row["median_sales"]:.0f}',
            va='center', fontsize=9, color='black')

plt.tight_layout()
plt.show()


# ANOVA - SALES BY COMPANY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/kpop_physical_sales.csv")
anova_df = df.dropna(subset=["parent_company", "sales", "annual_awards"])

plt.figure(figsize=(12, 7))
ax = sns.boxplot(data=anova_df, x="parent_company", y="sales")
plt.title("Sales by Parent Company")

from scipy.stats import f_oneway
f_sales, p_sales = f_oneway(*[group["sales"].values for name, group in anova_df.groupby("parent_company")])
anova_text = f"ANOVA: F = {f_sales:.2f}, p = {p_sales:.3f}"

plt.text(0.95, 0.95, anova_text,
         horizontalalignment='right',
         verticalalignment='top',
         transform=plt.gca().transAxes,
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ANOVA BY GROUP TYPE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/kpop_physical_sales.csv")
anova_df = df.dropna(subset=["group_type", "sales", "annual_awards"])

plt.figure(figsize=(8, 6))
ax = sns.boxplot(data=anova_df, x="group_type", y="sales",
                 hue="group_type", palette='pastel', dodge=False, legend=False)
plt.title("Sales by Group Type")

from scipy.stats import f_oneway
f_sales, p_sales = f_oneway(*[group["sales"].values for name, group in anova_df.groupby("group_type")])
anova_text = f"ANOVA: F = {f_sales:.2f}, p = {p_sales:.3f}"

plt.text(0.95, 0.95, anova_text,
         horizontalalignment='right',
         verticalalignment='top',
         transform=plt.gca().transAxes,
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
