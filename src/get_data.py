# DV - PHYSICAL ALBUM SALES
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://soridata.com/physical_sales.html?rank=sales&gto=1&gendero=0"

def scrape_physical_sales_colab(url=URL, chrome_path="/usr/bin/google-chrome-stable"):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    chrome_options.binary_location = chrome_path

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    try:
        driver.get(url)
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")

        if table is None:
            raise ValueError("Could not find any table on the page.")

        df_list = pd.read_html(str(table))
        if not df_list:
            raise ValueError("pandas.read_html returned no tables.")

        df = df_list[0]

        df = df.rename(columns={
            df.columns[0]: "rank",
            df.columns[1]: "artist",
            df.columns[2]: "sales"
        })

        df["sales"] = (
            df["sales"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.extract(r"(\d+)", expand=False)
            .astype(int)
        )

        return df

    finally:
        driver.quit()

df_sales = scrape_physical_sales_colab()
print(df_sales.head(20))
df_sales.to_csv("data/raw/kpop_physical_sales.csv", index=False, encoding="utf-8-sig")
print("Saved to kpop_physical_sales.csv")

# IV1 - PARENT COMPANY LABEL
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

target_groups = [
    "Stray Kids", "TXT", "Enhypen", "ATEEZ", "IVE", "Aespa", "NewJeans",
    "I-dle", "The Boyz", "Nmixx", "ITZY", "LE SSERAFIM",
    "Treasure", "CRAVITY", "WayV", "StayC", "Kep1er",
    "P1Harmony", "Oneus", "xikers"
]

wiki_name_corrections = {
    "TXT": "Tomorrow_X_Together",
    "I-dle": "(G)I-dle",
    "LE SSERAFIM": "LE_SSERAFIM",
    "StayC": "STAYC",
    "Oneus": "ONEUS",
}

parent_company_map = {
    "Belift Lab": "HYBE",
    "SOURCE MUSIC": "HYBE",
    "Source": "HYBE",
    "Source Music": "HYBE",
    "Big Hit Entertainment": "HYBE",
    "Big Hit": "HYBE",
    "ADOR": "HYBE",
    "HYBE": "HYBE",
    "SM Entertainment": "SM",
    "JYP Entertainment": "JYP",
    "YG Entertainment": "YG",
    "Starship Entertainment": "Starship",
    "Kakao M": "Kakao M",
    "Kakao Entertainment": "Kakao M",
    "RBW": "RBW",
    "Wake One": "CJ ENM",
    "CJ ENM": "CJ ENM",
    "HYBE Labels": "HYBE",
    "Cube": "Cube",
    "P NATION": "P NATION",
}

def get_wikipedia_url(group):
    page_name = wiki_name_corrections.get(group)
    if not page_name:
        page_name = group.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{page_name}"

def clean_company_name(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("\n", " ").strip()
    parts = text.split(",")
    company = parts[0].strip()
    company_upper = company.upper()
    for key in parent_company_map:
        if company_upper == key.upper():
            return parent_company_map[key]
    return company

def scrape_parent_company(group):
    url = get_wikipedia_url(group)
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            print(f"Failed to get Wikipedia page for {group} ({url}), status: {res.status_code}")
            return None

        soup = BeautifulSoup(res.text, "html.parser")
        infobox = soup.find("table", class_="infobox")

        if infobox is None:
            print(f"No infobox found on Wikipedia page for {group}")
            return None

        company_name = None
        for row in infobox.find_all("tr"):
            header = row.find("th")
            if header and header.text.strip().lower() in ["label", "labels", "parent company", "parent companies"]:
                td = row.find("td")
                if td:
                    company_name = clean_company_name(td.get_text(separator=","))
                    break

        if not company_name:
            print(f"Could not find label/parent company info for {group}")
            return None

        return company_name
    except Exception as e:
        print(f"Error scraping {group}: {e}")
        return None

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r"[^\w]", "", name)
    return name

def main():
    try:
        df = pd.read_csv("kpop_physical_sales.csv")
    except FileNotFoundError:
        print("kpop_physical_sales.csv not found!")
        return

    target_norm = [normalize_name(g) for g in target_groups]

    df['artist_norm'] = df['artist'].apply(normalize_name)

    df_filtered = df[df['artist_norm'].isin(target_norm)].copy()

    norm_to_group = dict(zip(target_norm, target_groups))

    group_to_company = {}
    for norm_name in target_norm:
        group = norm_to_group[norm_name]
        print(f"Scraping company for {group} ...")
        company = scrape_parent_company(group)
        print(f" -> {company}")
        group_to_company[group] = company
        time.sleep(2)

    df_filtered['parent_company'] = None

    for idx, row in df_filtered.iterrows():
        artist_norm = row['artist_norm']
        group = norm_to_group.get(artist_norm)
        if group:
            company = group_to_company.get(group)
            if company:
                df_filtered.at[idx, 'parent_company'] = company

    df_filtered.drop(columns=['artist_norm'], inplace=True)

    df_filtered.to_csv("data/raw/kpop_physical_sales.csv", index=False, encoding="utf-8-sig")
    print("\nUpdated data/raw/kpop_physical_sales.csv saved.")


if __name__ == "__main__":
    main()

# IV2 - GROUP TYPE
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

group_list = [
    "Stray Kids", "TXT", "Enhypen", "ATEEZ", "IVE", "Aespa", "NewJeans",
    "I-dle", "The Boyz", "Nmixx", "ITZY", "LE SSERAFIM",
    "Treasure", "CRAVITY", "WayV", "StayC", "Kep1er",
    "P1Harmony", "Oneus", "xikers"
]

wiki_name_corrections = {
    "TXT": "Tomorrow_X_Together",
    "I-dle": "(G)I-dle",
    "LE SSERAFIM": "LE_SSERAFIM",
    "StayC": "STAYC",
    "Oneus": "ONEUS",
}

def get_group_type_from_wikipedia(group_name):
    page_name = wiki_name_corrections.get(group_name, group_name.replace(" ", "_"))
    page_name = page_name.replace("-", "")

    url = f"https://en.wikipedia.org/wiki/{page_name}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"Wikipedia page not found or blocked for {group_name}: status {resp.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching page for {group_name}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    p = soup.find("p")
    if p and p.text:
        text = p.text.lower()
        if any(k in text for k in ["boy band", "boy group"]):
            return "boy"
        if any(k in text for k in ["girl band", "girl group"]):
            return "girl"

    cat_div = soup.find("div", id="mw-normal-catlinks")
    if cat_div:
        cats = cat_div.get_text().lower()
        if "boy bands" in cats or "boy groups" in cats:
            return "boy"
        if "girl groups" in cats or "girl bands" in cats:
            return "girl"

    print(f"Could not determine group type for {group_name}")
    return None

def main():
    df = pd.read_csv("data/raw/kpop_physical_sales.csv", dtype=str)  # updated path
    df["artist_norm"] = df["artist"].str.strip()

    group_type_map = {}
    for group in group_list:
        print(f"Scraping group type for {group} ...")
        group_type_map[group] = get_group_type_from_wikipedia(group)
        print(f" -> {group_type_map[group]}")
        time.sleep(1)

    df["group_type"] = df["artist_norm"].map(lambda x: group_type_map.get(x, None))

    df.drop(columns=["artist_norm"], inplace=True)

    df.to_csv("data/raw/kpop_physical_sales.csv", index=False, encoding="utf-8-sig")  # updated path
    print("Updated data/raw/kpop_physical_sales.csv with group_type column.")

if __name__ == "__main__":
    main()

# IV3 - DIGITAL SCORE
import pandas as pd
import requests

API_URL = "https://soridata.com/api/rank/kpop_streaming?rank=score&gto=1&gendero=0"

target_groups = [
    "Stray Kids", "TXT", "Enhypen", "ATEEZ", "IVE", "Aespa", "NewJeans",
    "I-dle", "The Boyz", "Nmixx", "ITZY", "LE SSERAFIM",
    "Treasure", "CRAVITY", "WayV", "StayC", "Kep1er",
    "P1Harmony", "Oneus", "xikers"
]

def scrape_digital_scores(api_url=API_URL):
    resp = requests.get(api_url)
    resp.raise_for_status()

    data = resp.json()

    if "data" not in data:
        raise ValueError("Unexpected API structure: 'data' not found")

    artists = []
    scores = []

    for item in data["data"]:
        artist = item.get("artist", "").strip()
        score = item.get("score")

        artists.append(artist)
        scores.append(score)

    df_scores = pd.DataFrame({
        "artist": artists,
        "score": scores
    })

    df_filtered = df_scores[df_scores["artist"].isin(target_groups)].copy()

    return df_filtered


def main():
    df_sales = pd.read_csv("data/raw/kpop_physical_sales.csv", dtype=str)  # updated input path

    df_digital_scores = scrape_digital_scores()

    df_sales["artist_norm"] = df_sales["artist"].str.strip()
    df_digital_scores["artist_norm"] = df_digital_scores["artist"].str.strip()

    df_merged = pd.merge(
        df_sales,
        df_digital_scores[["artist_norm", "score"]],
        on="artist_norm",
        how="left"
    )

    df_merged = df_merged.rename(columns={"score": "digital_score"})
    df_merged.drop(columns=["artist_norm"], inplace=True)

    df_merged.to_csv("data/raw/kpop_physical_sales.csv", index=False, encoding="utf-8-sig")  # updated output path

    print("Updated data/raw/kpop_physical_sales.csv with SoriData digital scores (blank if unavailable).")


if __name__ == "__main__":
    main()

# IV4 - YOUTUBE TOTAL VIEWS
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

target_groups = [
    "Stray Kids", "TXT", "Enhypen", "ATEEZ", "IVE", "Aespa", "NewJeans",
    "I-dle", "The Boyz", "Nmixx", "ITZY", "LE SSERAFIM",
    "Treasure", "CRAVITY", "WayV", "StayC", "Kep1er",
    "P1Harmony", "Oneus", "xikers"
]

special_name_map = {
    "(G)I-DLE": "idlee",
    "I-dle": "idlee",
    "Tomorrow X Together": "txt",
    "TXT": "txt",
    "NewJeans": "newjeans",
    "LE SSERAFIM": "lesserafim",
    "P1Harmony": "p1harmony",
    "StayC": "stayc",
    "WayV": "wayv",
    "The Boyz": "theboyz",
}

def normalize_name(name):
    if name in special_name_map:
        return special_name_map[name]
    return re.sub(r'\W+', '', name.lower())

def scrape_youtube_total_views():
    url = "https://kworb.net/youtube/archive.html"
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if not table:
        raise ValueError("Could not find YouTube views table")

    df = pd.read_html(str(table))[0]

    if "Artist" not in df.columns or "Total" not in df.columns:
        raise ValueError("Expected columns not found in the table")

    df['artist_norm'] = df['Artist'].apply(normalize_name)

    target_norm = [normalize_name(g) for g in target_groups]
    df_filtered = df[df['artist_norm'].isin(target_norm)]

    views_dict = dict(zip(df_filtered['artist_norm'], df_filtered['Total']))

    return views_dict

def main():
    sales_df = pd.read_csv("data/raw/kpop_physical_sales.csv", dtype=str)  # updated input path

    sales_df['artist_norm'] = sales_df['artist'].apply(normalize_name)

    youtube_views = scrape_youtube_total_views()

    youtube_view_list = []
    for artist_norm in sales_df['artist_norm']:
        views = youtube_views.get(artist_norm)
        if views is None:
            youtube_view_list.append("")
        else:
            youtube_view_list.append(views)

    sales_df['youtube_total_views'] = youtube_view_list
    sales_df.drop(columns=['artist_norm'], inplace=True)

    sales_df.to_csv("data/raw/kpop_physical_sales.csv", index=False, encoding='utf-8-sig')  # updated output path

    print("Updated data/raw/kpop_physical_sales.csv with youtube_total_views column")

if __name__ == "__main__":
    main()

# IV5 - ANNUAL AWARDS
import requests
import pandas as pd
from bs4 import BeautifulSoup

URL = "https://soridata.com/en/yawards.html?gto=1&gendero=0"

group_name_variants = {
    "stray kids": ["stray kids", "straykids"],
    "txt": ["txt", "tomorrow x together", "tomorrow by together"],
    "enhypen": ["enhypen"],
    "ateez": ["ateez"],
    "ive": ["ive"],
    "aespa": ["aespa"],
    "newjeans": ["newjeans", "new jeans"],
    "i-dle": ["i-dle", "(g)i-dle", "gidle"],
    "the boyz": ["the boyz", "theboyz"],
    "nmixx": ["nmixx"],
    "itzy": ["itzy"],
    "le sserafim": ["le sserafim", "lesserafim", "le-sserafim"],
    "treasure": ["treasure"],
    "cravity": ["cravity"],
    "wayv": ["wayv"],
    "stayc": ["stayc"],
    "kep1er": ["kep1er", "keppler"],
    "p1harmony": ["p1harmony", "p1h"],
    "oneus": ["oneus"],
    "xikers": ["xikers"]
}

def scrape_annual_awards(url=URL):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError("Could not find awards table on page")

    df_list = pd.read_html(str(table))
    if len(df_list) == 0:
        raise ValueError("Could not parse awards table")

    df_awards = df_list[0]

    df_awards.columns = [col.lower().strip() for col in df_awards.columns]
    if "artist" not in df_awards.columns or "awards" not in df_awards.columns:
        raise ValueError("Expected columns 'artist' and 'awards' not found")

    df_awards["artist_norm"] = df_awards["artist"].astype(str).str.strip().lower()

    results = {}

    for group_key, variants in group_name_variants.items():
        matched_rows = None
        for variant in variants:
            v = variant.lower().strip()
            matched_rows = df_awards[df_awards["artist_norm"].str.contains(v, na=False)]
            if not matched_rows.empty:
                break

        if matched_rows is not None and not matched_rows.empty:
            val = matched_rows.iloc[0]["awards"]
            try:
                val = int(str(val).replace(",", "").strip())
            except:
                val = None
            results[group_key] = val
        else:
            results[group_key] = None

    df_results = pd.DataFrame.from_dict(results, orient="index", columns=["annual_awards"])
    df_results.index.name = "group"
    df_results.reset_index(inplace=True)

    return df_results


if __name__ == "__main__":
    df_awards = scrape_annual_awards()
    print(df_awards)

    try:
        df_sales = pd.read_csv("data/raw/kpop_physical_sales.csv")  # updated input path

        df_sales["group_norm"] = df_sales["artist"].str.lower().str.replace(" ", "").str.replace("-", "")
        df_awards["group_norm"] = df_awards["group"].str.lower().str.replace(" ", "").str.replace("-", "")

        df_merged = pd.merge(df_sales, df_awards[["group_norm", "annual_awards"]], on="group_norm", how="left")

        df_merged.drop(columns=["group_norm"], inplace=True)
        df_merged.to_csv("data/raw/kpop_physical_sales.csv", index=False, encoding="utf-8-sig")  # updated output path

        print("\nUpdated data/raw/kpop_physical_sales.csv with annual awards for 20 groups.")

    except FileNotFoundError:
        print("ERROR: data/raw/kpop_physical_sales.csv not found.")

# IV6 - MUSIC SHOW AWARDS
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

group_aliases = {
    "stray kids": ["stray kids", "straykids"],
    "txt": ["txt", "tomorrow x together", "tomorrow by together"],
    "enhypen": ["enhypen"],
    "ateez": ["ateez"],
    "ive": ["ive"],
    "aespa": ["aespa"],
    "newjeans": ["newjeans", "new jeans"],
    "i-dle": ["i-dle", "(g)i-dle", "gidle"],
    "the boyz": ["the boyz", "theboyz"],
    "nmixx": ["nmixx"],
    "itzy": ["itzy"],
    "le sserafim": ["le sserafim", "lesserafim", "le-sserafim"],
    "treasure": ["treasure"],
    "cravity": ["cravity"],
    "wayv": ["wayv"],
    "stayc": ["stayc"],
    "kep1er": ["kep1er", "keppler"],
    "p1harmony": ["p1harmony", "p1h"],
    "oneus": ["oneus"],
    "xikers": ["xikers"]
}

def scrape_music_show_awards():
    url = "https://kpopping.com/musicshows/total-wins"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    if table is None:
        raise ValueError("Could not find table on KPopping page")

    df = pd.read_html(StringIO(str(table)))[0]

    df.columns = [col.lower() for col in df.columns]

    df.rename(columns={"winner": "group"}, inplace=True)

    df["group"] = df["group"].astype(str).str.lower()
    df["wins"] = pd.to_numeric(df["wins"], errors="coerce").fillna(0).astype(int)

    return df


def match_group_awards(scraped_df):
    result = {}

    for canonical, alias_list in group_aliases.items():
        found = 0
        for alias in alias_list:
            alias = alias.lower()
            match = scraped_df[scraped_df["group"] == alias]
            if not match.empty:
                found = int(match["wins"].iloc[0])
                break
        result[canonical] = found

    return result


def update_csv_with_awards(csv_path="data/raw/kpop_physical_sales.csv"):  # updated default path
    scraped_df = scrape_music_show_awards()
    awards_dict = match_group_awards(scraped_df)

    df_existing = pd.read_csv(csv_path)

    if "artist" not in df_existing.columns:
        raise ValueError("Your CSV does not contain 'artist' colum

# IV7 - YOUTUBE SUBSCRIBERS
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from io import StringIO

URL = "https://kpopping.com/database/group/youtube-followers"

target_groups = [
    "Stray Kids", "TXT", "Enhypen", "ATEEZ", "IVE", "Aespa", "NewJeans",
    "I-dle", "The Boyz", "Nmixx", "ITZY", "LE SSERAFIM",
    "Treasure", "CRAVITY", "WayV", "StayC", "Kep1er",
    "P1Harmony", "Oneus", "xikers"
]

alias_map = {
    "stray kids": ["stray kids", "straykids"],
    "txt": ["txt", "tomorrow x together", "tomorrow by together"],
    "enhypen": ["enhypen"],
    "ateez": ["ateez"],
    "ive": ["ive"],
    "aespa": ["aespa"],
    "newjeans": ["newjeans", "new jeans"],
    "i-dle": ["i-dle", "(g)i-dle", "gidle"],
    "the boyz": ["the boyz", "theboyz"],
    "nmixx": ["nmixx"],
    "itzy": ["itzy"],
    "le sserafim": ["le sserafim", "lesserafim", "le-sserafim"],
    "treasure": ["treasure"],
    "cravity": ["cravity"],
    "wayv": ["wayv"],
    "stayc": ["stayc"],
    "kep1er": ["kep1er", "keppler"],
    "p1harmony": ["p1harmony", "p1h"],
    "oneus": ["oneus"],
    "xikers": ["xikers"]
}

def normalize(s: str) -> str:
    """Lowercase, remove non-alphanumeric for matching."""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def scrape_youtube_subscribers():
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(URL, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError("Could not find subscribers table on kpopping page")

    df = pd.read_html(StringIO(str(table)))[0]
    df.columns = [c.lower().strip() for c in df.columns]

    if "name" not in df.columns or "subscribers" not in df.columns:
        raise ValueError("Expected columns 'name' and 'subscribers' not found")

    df = df[["name", "subscribers"]].copy()
    df["name_norm"] = df["name"].astype(str).str.strip().apply(normalize)

    df["subscribers"] = (
        df["subscribers"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
    )
    df["subscribers"] = pd.to_numeric(df["subscribers"], errors="coerce").fillna(0).astype(int)

    return df

def get_subscribers_for_group(group_name: str, df_subs: pd.DataFrame):
    """Find YouTube subscriber count using alias matching."""
    key = group_name.lower().strip()
    possible = alias_map.get(key, [key])

    for alias in possible:
        norm = normalize(alias)
        match = df_subs[df_subs["name_norm"] == norm]
        if not match.empty:
            return int(match.iloc[0]["subscribers"])

    return None

def update_csv_with_youtube(csv_path="data/raw/kpop_physical_sales.csv"):  # updated path here
    df_subs = scrape_youtube_subscribers()
    df_sales = pd.read_csv(csv_path, dtype=str)

    if "artist" not in df_sales.columns:
        raise ValueError("CSV must contain an 'artist' column")

    subscribers_list = []
    for artist in df_sales["artist"]:
        value = get_subscribers_for_group(artist, df_subs)
        subscribers_list.append(value if value is not None else "")

    df_sales["youtube_subscribers"] = subscribers_list

    df_sales.to_csv(csv_path, index=False, encoding="utf-8-sig")  # updated path here
    print("Updated CSV with youtube_subscribers column.")
    return df_sales

if __name__ == "__main__":
    updated = update_csv_with_youtube()
    print(updated.head(20))

# IV8 - DAYS UNTIL FIRST WIN

import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from io import StringIO

URL = "https://kpopping.com/database/group/days-until-1st-win"

group_aliases = {
    "stray kids": ["stray kids", "straykids", "스트레이 키즈"],
    "txt": ["txt", "tomorrow x together", "tomorrow by together"],
    "enhypen": ["enhypen"],
    "ateez": ["ateez"],
    "ive": ["ive"],
    "aespa": ["aespa"],
    "newjeans": ["newjeans", "new jeans"],
    "i-dle": ["i-dle", "(g)i-dle", "gidle", "아이들"],
    "the boyz": ["the boyz", "theboyz", "더보이즈"],
    "nmixx": ["nmixx"],
    "itzy": ["itzy"],
    "le sserafim": ["le sserafim", "lesserafim", "le-sserafim", "르세라핌"],
    "treasure": ["treasure"],
    "cravity": ["cravity"],
    "wayv": ["wayv"],
    "stayc": ["stayc"],
    "kep1er": ["kep1er", "keppler"],
    "p1harmony": ["p1harmony", "p1h"],
    "oneus": ["oneus"],
    "xikers": ["xikers"]
}

def normalize_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

def scrape_days_to_first_win():
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(URL, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError("Could not find table on kpopping days-until-1st-win page")

    df = pd.read_html(StringIO(str(table)))[0]
    df.columns = [c.strip().lower() for c in df.columns]

    col_name = None
    col_days = None
    for col in df.columns:
        if "name" in col or "group" in col:
            col_name = col
        if "days" in col and ("first" in col or "1st" in col or "win" in col):
            col_days = col

    if not col_name or not col_days:
        raise ValueError(f"Could not find expected columns. Found: {df.columns}")

    df = df[[col_name, col_days]].copy()
    df = df.rename(columns={col_name: "group", col_days: "days_til_first_win"})

    df["group_norm"] = df["group"].astype(str).str.strip().apply(normalize_name)
    df["days_til_first_win"] = pd.to_numeric(
        df["days_til_first_win"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    return df

def get_days_for_group(artist: str, df_days: pd.DataFrame):
    artist_norm = normalize_name(artist)  # FIX: added this

    for canonical, variants in group_aliases.items():
        variant_norms = [normalize_name(v) for v in variants]

        if artist_norm in variant_norms:
            canonical_norm = normalize_name(canonical)
            match = df_days[df_days["group_norm"] == canonical_norm]

            if not match.empty:
                val = match.iloc[0]["days_til_first_win"]
                if pd.notna(val):
                    return int(val)

    return None

def update_csv(csv_path="data/raw/kpop_physical_sales.csv"):  # updated path here
    df_days = scrape_days_to_first_win()
    df = pd.read_csv(csv_path, dtype=str)

    if "artist" not in df.columns:
        raise ValueError("CSV must include an 'artist' column")

    result = []
    for artist in df["artist"]:
        val = get_days_for_group(artist, df_days)
        result.append(val if val is not None else "")

    df["days_to_first_win"] = result
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # updated path here

    print("Updated CSV with days_to_first_win column.")
    return df


if __name__ == "__main__":
    df_updated = update_csv()
    print(df_updated.head())
