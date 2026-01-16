import pandas as pd
from pathlib import Path

# Load data
project_dir = Path(__file__).parent.absolute()
csv_path = project_dir / "fbref_cleaned_2425.csv"

if not csv_path.exists():
    print("❌ CSV not found.")
    exit()

df = pd.read_csv(csv_path, index_col=[0, 1, 2, 3])
cols = list(df.columns)

print("--- GK STAT AUDIT ---")

def check(keyword):
    matches = [c for c in cols if keyword.lower() in c.lower()]
    print(f"Searching '{keyword}': {matches}")

check("PSxG")   # Expected Goals (Shot Stopping Skill)
check("Stp")    # Crosses Stopped (Aerial Command)
check("OPA")    # Defensive Actions Outside Box (Sweeper Keeper Skill)
check("Launch") # Long Passing