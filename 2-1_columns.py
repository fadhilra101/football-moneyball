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

print("--- COLUMN HUNTER ---")

def find_col(keyword):
    matches = [c for c in cols if keyword.lower() in c.lower()]
    print(f"\nSearching for '{keyword}':")
    for m in matches:
        print(f"  Found: {m}")

# Search for the missing MF/DF stats
find_col("Prg")       # For Progressive Passes/Carries
find_col("Tkl")       # For Tackles
find_col("Int")       # For Interceptions
find_col("Blocks")    # For Blocks
find_col("Cmp%")      # For Completion %
find_col("Final")     # For Final Third passes
find_col("1/3")       # Alternative for Final Third