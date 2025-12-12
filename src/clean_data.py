# IV1 - PARENT COMPANY MISSING VALUE
import pandas as pd

df = pd.read_csv("data/raw/kpop_physical_sales.csv")

manual_overrides = {
    "ATEEZ": "KQ",
    "IVE": "Starship",
    "NewJeans": "HYBE",
    "The Boyz": "IST",
    "ITZY": "JYP",
    "Treasure": "YG",
    "Oneus": "RBW"
}

df['artist_norm'] = df['artist'].str.lower().str.replace(r'\W+', '', regex=True)

for group, company in manual_overrides.items():
    group_norm = group.lower().replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
    mask = df['artist_norm'] == group_norm
    df.loc[mask, 'parent_company'] = company

df = df.drop(columns=['artist_norm'])

df.to_csv("data/processed/kpop_physical_sales.csv", index=False, encoding="utf-8-sig")

print("Updated parent companies manually and saved to data/processed/kpop_physical_sales.csv")

# IV2 - GT MISSING VALUE
import pandas as pd

df = pd.read_csv("data/raw/kpop_physical_sales.csv", dtype=str)

df["artist_norm"] = df["artist"].str.strip()

manual_updates = {
    "IVE": "girl",
    "I-dle": "girl",
    "The Boyz": "boy",
    "Treasure": "boy",
    "Oneus": "boy"
}

for group, gender in manual_updates.items():
    mask = df["artist_norm"] == group
    df.loc[mask, "group_type"] = gender

df.drop(columns=["artist_norm"], inplace=True)

df.to_csv("data/processed/kpop_physical_sales.csv", index=False, encoding="utf-8-sig")

print("Manual group_type updates for selected groups completed and saved to data/processed/kpop_physical_sales.csv")
